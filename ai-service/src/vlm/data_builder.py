import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.dataset import NIH_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class VLMInstructionItem:
    """Standardized multimodal instruction record."""
    image_path: str
    patient_id: str
    prompt: str
    response: str
    task_type: str  # 'findings_list', 'yes_no_query', 'structured_summary'
    ground_truth_labels: List[str]


def build_vlm_instruction_dataset(
    records: List[Dict[str, Any]],
    classes: Optional[List[str]] = None,
    task_distribution: Optional[Dict[str, float]] = None,
) -> List[VLMInstructionItem]:
    """
    Builds instruction-style VLM datasets directly from ground truth NIH labels.
    Never fabricates free-form radiology narratives.
    
    Generates 3 clinical instruction formats:
      1. 'findings_list': lists active thoracic abnormalities or 'No Finding'
      2. 'yes_no_query': asks about a specific condition (clinical question answering)
      3. 'structured_summary': systematic checklist over all 14 findings
    """
    if classes is None:
        classes = NIH_CLASSES

    instruction_items: List[VLMInstructionItem] = []

    for rec in records:
        img_path = rec["path"]
        patient_id = rec.get("patient_id", "")
        raw_labels = rec.get("labels", [])
        norm_labels = [l.replace(" ", "_") for l in raw_labels if l in classes or l.replace(" ", "_") in classes]

        # 1. Findings List Instruction
        if norm_labels:
            resp_list = f"Identified findings: {', '.join(sorted(norm_labels))}."
        else:
            resp_list = "No Finding. The chest radiograph demonstrates no acute thoracic abnormalities."

        instruction_items.append(
            VLMInstructionItem(
                image_path=img_path,
                patient_id=patient_id,
                prompt="Examine this frontal chest radiograph and list all identified thoracic abnormalities.",
                response=resp_list,
                task_type="findings_list",
                ground_truth_labels=norm_labels,
            )
        )

        # 2. Targeted Query (Yes/No Question for 2 representative classes per image)
        for target_cls in [classes[0], classes[2]]:  # Atelectasis, Effusion
            is_present = target_cls in norm_labels
            resp_qa = (
                f"Yes, findings consistent with {target_cls} are present."
                if is_present
                else f"No, no evidence of {target_cls} is observed."
            )
            instruction_items.append(
                VLMInstructionItem(
                    image_path=img_path,
                    patient_id=patient_id,
                    prompt=f"Does this chest X-ray show evidence of {target_cls}?",
                    response=resp_qa,
                    task_type="yes_no_query",
                    ground_truth_labels=norm_labels,
                )
            )

        # 3. Structured Multi-Label Checklist
        lines = []
        for cls in classes:
            status = "Positive" if cls in norm_labels else "Negative"
            lines.append(f"- {cls}: {status}")
        summary_resp = "Diagnostic Summary:\n" + "\n".join(lines)

        instruction_items.append(
            VLMInstructionItem(
                image_path=img_path,
                patient_id=patient_id,
                prompt="Provide a systematic multi-label diagnostic summary for the 14 standard thoracic conditions.",
                response=summary_resp,
                task_type="structured_summary",
                ground_truth_labels=norm_labels,
            )
        )

    logger.info("Generated %d instruction items from %d image records", len(instruction_items), len(records))
    return instruction_items


def save_vlm_dataset_jsonl(items: List[VLMInstructionItem], output_file: str | Path) -> Path:
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
    logger.info("Saved VLM instruction dataset to %s", out)
    return out
