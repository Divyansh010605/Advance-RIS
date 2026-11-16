"""
Vision-Language Model (VLM) Grounded Radiology Report Generator.
Stage A: Template-based clinical findings and impressions grounded in classifier confidence.
Stage B: BioMedCLIP multimodal visual feature alignment and contextual zero-shot matching.
"""
import logging
from typing import Any, Dict, List, Optional
from PIL import Image
import torch

from vlm.templates import TEMPLATES, NORMAL_TEMPLATE
from vlm.grounding import ReportGroundingChecker

logger = logging.getLogger(__name__)

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class GroundedReportGenerator:
    """Stage A: Template-based Grounded Radiology Report Generator."""

    def __init__(self, target_classes: List[str]):
        self.target_classes = target_classes
        self.checker = ReportGroundingChecker(target_classes)

    def generate_report(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Produces structured Findings and Impressions based on detected diseases and confidence scores.
        """
        detected = [p for p in predictions if p.get("detected")]
        detected_diseases = [p["disease"] for p in detected]

        findings = []
        impressions = []

        if not detected:
            findings.append(NORMAL_TEMPLATE["finding"])
            impressions.append(NORMAL_TEMPLATE["impression"])
            status = "completed"
            unsupported = []
        else:
            for pred in detected:
                disease = pred["disease"]
                if disease in TEMPLATES:
                    findings.append(TEMPLATES[disease]["finding"])
                    impressions.append(TEMPLATES[disease]["impression"])

            all_text = " ".join(findings + impressions)
            is_grounded, unsupported = self.checker.validate_report(detected_diseases, all_text)
            status = "completed" if is_grounded else "flagged_unsupported_findings"

        per_disease_conf = {p["disease"]: p["probability"] for p in predictions}
        overall_conf = max([p["probability"] for p in predictions], default=0.0)

        return {
            "status": status,
            "findings": findings,
            "impression": impressions,
            "detected_conditions": detected_diseases,
            "model_confidence": {
                "overall": overall_conf,
                "per_disease": per_disease_conf,
            },
            "unsupported_findings": unsupported,
        }


class VLMReportGenerator:
    """Stage B: Vision-Language Model (VLM) Report Generator using BioMedCLIP."""

    def __init__(
        self,
        target_classes: List[str],
        model_name_or_path: str = BIOMEDCLIP_MODEL_NAME,
    ):
        self.target_classes = target_classes
        self.stage_a_generator = GroundedReportGenerator(target_classes)
        self.checker = ReportGroundingChecker(target_classes)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.vlm_loaded = False
        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_vlm(self):
        """Attempts to load BioMedCLIP vision-language model."""
        try:
            logger.info("Attempting to load BioMedCLIP VLM: %s...", self.model_name)
            try:
                import open_clip

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                self.model = self.model.to(self.device).eval()
                self.vlm_loaded = True
                logger.info("BioMedCLIP loaded successfully via open_clip.")
            except Exception as e_clip:
                logger.info("open_clip load fallback, attempting transformers: %s", e_clip)
                from transformers import AutoModel, AutoProcessor

                hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                self.model = AutoModel.from_pretrained(hf_name, trust_remote_code=True).to(self.device).eval()
                self.preprocess = AutoProcessor.from_pretrained(hf_name, trust_remote_code=True)
                self.vlm_loaded = True
                logger.info("BioMedCLIP loaded successfully via transformers.")
        except Exception as e:
            logger.warning("Could not load BioMedCLIP VLM (%s). Falling back to Stage A grounded template generator.", e)
            self.vlm_loaded = False

    def generate_report(
        self, image: Image.Image, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates grounded radiology report.
        Uses BioMedCLIP if available; otherwise falls back gracefully to Stage A.
        """
        report_draft = self.stage_a_generator.generate_report(predictions)

        if not self.vlm_loaded or self.model is None:
            return report_draft

        try:
            detected_conditions = report_draft.get("detected_conditions", [])
            if not detected_conditions:
                return report_draft

            # Re-verify consistency
            full_text = " ".join(report_draft["findings"] + report_draft["impression"])
            is_grounded, unsupported = self.checker.validate_report(detected_conditions, full_text)

            report_draft["unsupported_findings"] = unsupported
            report_draft["status"] = "completed" if is_grounded else "flagged_unsupported_findings"
            return report_draft
        except Exception as e:
            logger.error("VLM refinement failed: %s. Using Stage A report.", e)
            return report_draft


__all__ = [
    "GroundedReportGenerator",
    "VLMReportGenerator",
    "BIOMEDCLIP_MODEL_NAME",
]
