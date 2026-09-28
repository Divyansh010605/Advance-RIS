"""
Template-based Grounded Radiology Report Generator (Stage A).
Produces structured Findings and Impressions based on detected diseases and confidence scores.
"""
from typing import List, Dict, Any
from report_generation.report_grounding import ReportGroundingChecker

TEMPLATES: Dict[str, Dict[str, str]] = {
    "Tuberculosis": {
        "finding": "Opacities and apical cavitary infiltrates suggestive of active or prior Tuberculosis.",
        "impression": "Findings are suspicious for active pulmonary Tuberculosis."
    },
    "Sick_Non_TB": {
        "finding": "Non-tuberculosis pulmonary opacities, focal consolidations, or interstitial markings observed.",
        "impression": "Evidence of non-TB thoracic pulmonary disease (Sick_Non_TB)."
    },
    "Healthy": {
        "finding": "Lungs are clear bilaterally with normal cardiomediastinal silhouette and no focal consolidation.",
        "impression": "Normal chest X-ray examination (Healthy)."
    }
}



class GroundedReportGenerator:
    def __init__(self, target_classes: List[str]):
        self.checker = ReportGroundingChecker(target_classes)

    def generate_report(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a draft report from model prediction results.
        predictions is a list of dicts: [{'disease': str, 'probability': float, 'detected': bool}]
        """
        detected_items = [p for p in predictions if p.get("detected", False)]
        detected_names = [p["disease"] for p in detected_items]

        findings = []
        impression = []
        detected_conditions = []
        per_disease_conf = {}

        if not detected_items:
            findings.append("No acute focal pulmonary consolidation, effusion, or pneumothorax identified.")
            findings.append("Lungs are clear bilaterally with normal cardiomediasternal silhouette.")
            impression.append("No significant thoracic pathology detected.")
        else:
            for item in detected_items:
                disease = item["disease"]
                prob = item["probability"]
                per_disease_conf[disease] = round(prob, 4)

                detected_conditions.append({
                    "name": disease,
                    "confidence": round(prob, 4)
                })

                if disease in TEMPLATES:
                    findings.append(TEMPLATES[disease]["finding"])
                    impression.append(TEMPLATES[disease]["impression"])
                else:
                    findings.append(f"Abnormality identified matching {disease} pattern.")
                    impression.append(f"Evidence of {disease}.")

        overall_conf = round(sum(p["probability"] for p in predictions) / len(predictions), 4) if predictions else 0.0

        full_text = " ".join(findings + impression)
        is_grounded, unsupported = self.checker.validate_report(detected_names, full_text)

        report = {
            "status": "draft",
            "findings": findings,
            "impression": impression,
            "detected_conditions": detected_conditions,
            "model_confidence": {
                "overall": overall_conf,
                "per_disease": per_disease_conf
            },
            "review_required": True,
            "unsupported_findings": unsupported,
            "disclaimer": "AI-generated draft for research and educational purposes. Human review is required before finalization."
        }

        return report
