"""
VLM Evaluation: Parses structured findings from fine-tuned VLM outputs and
computes hallucination rate, precision, recall against NIH ground-truth labels.

Usage:
  python -m src.eval.vlm_eval --vlm medgemma --model_dir outputs/vlm/medgemma_lora \\
         --test_jsonl outputs/vlm/medgemma_lora/val_instructions.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.data.dataset import NIH_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.eval.vlm_eval")

NIH_CLASSES_LOWER: Set[str] = {c.lower().replace("_", " ") for c in NIH_CLASSES}


# ─── Findings Parser ──────────────────────────────────────────────────────────

def parse_findings_from_text(text: str) -> List[str]:
    """
    Extract NIH disease labels from free-form VLM output text.

    Strategies:
      1. Direct class name matching (case-insensitive).
      2. 'Positive'/'Negative' structured checklist parsing.
      3. Comma-separated findings list after 'Identified findings:'.

    Returns list of normalized NIH disease names found in text.
    """
    found: List[str] = []
    text_lower = text.lower()

    # Pre-compute set of classes explicitly marked Negative in checklist format
    explicit_negatives: Set[str] = set()
    for match in re.finditer(r"-\s*(.+?):\s*(positive|negative)", text_lower):
        label_raw, status = match.group(1).strip(), match.group(2).strip()
        if status == "negative":
            for cls in NIH_CLASSES:
                if cls.lower().replace("_", " ") == label_raw or cls.lower() == label_raw:
                    explicit_negatives.add(cls)

    # Strategy 1: Direct name matching (skip explicit negatives)
    for cls in NIH_CLASSES:
        if cls in explicit_negatives:
            continue
        cls_lower = cls.lower().replace("_", " ")
        if cls_lower in text_lower:
            # Verify it's NOT preceded by a negating phrase
            pattern = rf"(no evidence of|not|without|no )\s+{re.escape(cls_lower)}"
            if not re.search(pattern, text_lower):
                found.append(cls)

    # Strategy 2: Parse "- Disease: Positive" format
    for match in re.finditer(r"-\s*(.+?):\s*(positive|negative)", text_lower):
        label, status = match.group(1).strip(), match.group(2).strip()
        if status == "positive":
            # Normalize label → NIH class
            for cls in NIH_CLASSES:
                if cls.lower().replace("_", " ") == label or cls.lower() == label:
                    if cls not in found:
                        found.append(cls)

    # Strategy 3: "Identified findings: X, Y, Z"
    findings_match = re.search(r"identified findings:\s*(.+?)(?:\.|$)", text_lower)
    if findings_match:
        raw_list = findings_match.group(1)
        if "no finding" not in raw_list:
            for item in raw_list.split(","):
                item = item.strip()
                for cls in NIH_CLASSES:
                    if cls.lower().replace("_", " ") == item or cls.lower() == item:
                        if cls not in found:
                            found.append(cls)

    return list(set(found))


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def evaluate_vlm_predictions(
    generated_texts: List[str],
    ground_truth_labels: List[List[str]],
    classes: List[str] = NIH_CLASSES,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, and hallucination rate from VLM outputs.

    Args:
        generated_texts: List of raw VLM output strings.
        ground_truth_labels: List of ground truth NIH label lists per sample.

    Returns:
        Dict with macro precision, recall, F1, exact_match_rate, hallucination_rate.
    """
    assert len(generated_texts) == len(ground_truth_labels), "Mismatch in list lengths."

    total_tp = total_fp = total_fn = 0
    hallucination_count = 0
    exact_match_count = 0

    per_class_tp: Dict[str, int] = {c: 0 for c in classes}
    per_class_fp: Dict[str, int] = {c: 0 for c in classes}
    per_class_fn: Dict[str, int] = {c: 0 for c in classes}

    for gen, gt in zip(generated_texts, ground_truth_labels):
        pred_set = set(parse_findings_from_text(gen))
        gt_set = set(gt)

        tp = pred_set & gt_set
        fp = pred_set - gt_set
        fn = gt_set - pred_set

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        if fp:
            hallucination_count += 1
        if pred_set == gt_set:
            exact_match_count += 1

        for cls in classes:
            if cls in tp:
                per_class_tp[cls] += 1
            if cls in fp:
                per_class_fp[cls] += 1
            if cls in fn:
                per_class_fn[cls] += 1

    n = len(generated_texts)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Per-class F1
    per_class_f1 = {}
    for cls in classes:
        p = per_class_tp[cls] / max(1, per_class_tp[cls] + per_class_fp[cls])
        r = per_class_tp[cls] / max(1, per_class_tp[cls] + per_class_fn[cls])
        per_class_f1[cls] = 2 * p * r / max(1e-9, p + r)

    return {
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "exact_match_rate": round(exact_match_count / n, 4),
        "hallucination_rate": round(hallucination_count / n, 4),
        "total_samples": n,
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
    }


# ─── CLI Inference + Evaluation ──────────────────────────────────────────────

def run_inference_and_evaluate(
    model_dir: str,
    test_jsonl: str,
    vlm_key: str = "medgemma",
    max_new_tokens: int = 128,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Load a fine-tuned PEFT checkpoint and run inference over test items.
    Returns evaluation metrics dict.
    """
    import torch  # noqa
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore

    logger.info("Loading base model and PEFT adapter from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    device = next(model.parameters()).device

    # Load test items from JSONL
    items = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if num_samples:
        items = items[:num_samples]

    logger.info("Running inference on %d test items...", len(items))
    generated_texts = []
    ground_truth_labels = []

    for item in items:
        prompt = item["prompt"]
        gt_labels = item.get("ground_truth_labels", [])

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_texts.append(gen_text)
        ground_truth_labels.append(gt_labels)

    metrics = evaluate_vlm_predictions(generated_texts, ground_truth_labels)
    logger.info("Evaluation results: %s", json.dumps(metrics, indent=2))

    # Save results
    out_path = Path(model_dir) / "eval_results.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Evaluation — Findings Parsing & Hallucination Rate")
    parser.add_argument("--vlm", choices=["medgemma", "llava_med", "biomedgpt"], default="medgemma")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned PEFT adapter directory.")
    parser.add_argument("--test_jsonl", required=True, help="Path to val_instructions.jsonl.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Test parser only, no model inference.")
    args = parser.parse_args()

    if args.dry_run:
        # Test the findings parser
        test_texts = [
            "Identified findings: Atelectasis, Effusion.",
            "- Atelectasis: Positive\n- Cardiomegaly: Negative\n- Effusion: Positive",
            "No Finding. The chest radiograph demonstrates no acute thoracic abnormalities.",
        ]
        for t in test_texts:
            parsed = parse_findings_from_text(t)
            logger.info("Parsed: %s → %s", repr(t[:60]), parsed)
        logger.info("[DRY-RUN] VLM evaluation parser validated.")
        return

    metrics = run_inference_and_evaluate(
        model_dir=args.model_dir,
        test_jsonl=args.test_jsonl,
        vlm_key=args.vlm,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
