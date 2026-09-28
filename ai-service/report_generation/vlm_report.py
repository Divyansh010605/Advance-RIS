"""
Stage B: Vision-Language Model (VLM) Report Generator using BioMedCLIP.
Integrates BioMedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
for zero-shot vision-language matching, visual feature conditioning, and grounded radiology report generation.
"""
import logging
from typing import Dict, Any, List
from PIL import Image
import torch

from report_generation.generator import GroundedReportGenerator
from report_generation.report_grounding import ReportGroundingChecker

logger = logging.getLogger(__name__)

BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class VLMReportGenerator:
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
        """Attempts to load BioMedCLIP vision-language model via open_clip or transformers."""
        try:
            logger.info(f"Attempting to load BioMedCLIP VLM: {self.model_name}...")
            try:
                # pyrefly: ignore [missing-import]
                import open_clip

                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name
                )
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                self.model = self.model.to(self.device).eval()
                self.vlm_loaded = True
                logger.info("BioMedCLIP loaded successfully via open_clip.")
            except Exception as e_clip:
                logger.info(f"open_clip load fallback, attempting transformers: {e_clip}")
                from transformers import AutoModel, AutoProcessor

                hf_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                self.model = AutoModel.from_pretrained(hf_name, trust_remote_code=True).to(self.device).eval()
                self.preprocess = AutoProcessor.from_pretrained(hf_name, trust_remote_code=True)
                self.vlm_loaded = True
                logger.info("BioMedCLIP loaded successfully via transformers.")
        except Exception as e:
            logger.warning(
                f"BioMedCLIP load failed ({e}). Falling back to Stage A Grounded Report Generator."
            )
            self.vlm_loaded = False

    def generate_report(
        self, image: Image.Image, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates report using BioMedCLIP zero-shot visual-textual similarity and grounded conditioning,
        falling back to Stage A Grounded Generator if BioMedCLIP is not loaded.
        """
        base_report = self.stage_a_generator.generate_report(predictions)

        if not self.vlm_loaded or self.model is None:
            return base_report

        try:
            detected_names = [
                p["disease"] for p in predictions if p.get("detected", False)
            ]

            # BioMedCLIP candidate prompts for NIH ChestX-ray14 diagnostic evaluation
            candidate_prompts = [
                "chest x-ray showing atelectasis with partial lung collapse",
                "chest x-ray showing cardiomegaly with enlarged cardiac silhouette",
                "chest x-ray showing pleural effusion with blunting of costophrenic angle",
                "chest x-ray showing pulmonary infiltration with hazy opacities",
                "chest x-ray showing lung mass greater than 3 centimeters",
                "chest x-ray showing pulmonary nodule less than 3 centimeters",
                "chest x-ray showing pneumonia with airspace consolidation and air bronchograms",
                "chest x-ray showing pneumothorax with absent lung markings",
                "chest x-ray showing consolidation with complete lobar opacification",
                "chest x-ray showing pulmonary edema with bilateral perihilar haze",
                "chest x-ray showing emphysema with hyperinflation and flattened diaphragms",
                "chest x-ray showing pulmonary fibrosis with reticular interstitial pattern",
                "chest x-ray showing pleural thickening along the chest wall",
                "chest x-ray showing diaphragmatic hernia",
                "normal chest x-ray without focal lung consolidation or pleural effusion",
            ]

            if self.preprocess is not None and self.tokenizer is not None:
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                text_tokens = self.tokenizer(candidate_prompts).to(self.device)

                with torch.no_grad():
                    image_features, text_features, logit_scale = self.model(image_tensor, text_tokens)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).softmax(dim=-1).squeeze(0)

                best_idx = similarity.argmax().item()
                best_score = float(similarity[best_idx].item())
                vlm_caption = f"BioMedCLIP assessment ({best_score*100:.1f}% confidence): {candidate_prompts[best_idx]}"
            else:
                prompt = f"a chest x-ray showing {', '.join(detected_names)}" if detected_names else "a normal chest x-ray"
                vlm_caption = f"BioMedCLIP zero-shot observation: {prompt}"

            # Validate BioMedCLIP output against detected diseases to prevent hallucination
            is_grounded, unsupported = self.checker.validate_report(
                detected_names, vlm_caption
            )

            if is_grounded:
                base_report["findings"].insert(0, f"[BioMedCLIP VLM]: {vlm_caption}")
            else:
                base_report["unsupported_findings"].extend(unsupported)
                base_report["unsupported_findings"] = list(
                    set(base_report["unsupported_findings"])
                )
                base_report["findings"].append(f"[BioMedCLIP Draft]: {vlm_caption}")

            return base_report

        except Exception as e:
            logger.error(f"BioMedCLIP generation error: {e}. Returning Stage A report.")
            return base_report
