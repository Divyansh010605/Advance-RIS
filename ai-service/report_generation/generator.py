"""
Template-based Grounded Radiology Report Generator (Stage A).
Produces structured Findings and Impressions based on detected diseases and confidence scores.
"""
from typing import List, Dict, Any
from report_generation.report_grounding import ReportGroundingChecker

TEMPLATES: Dict[str, Dict[str, str]] = {
    "Atelectasis": {
        "finding": "Linear or plate-like opacity consistent with subsegmental atelectasis, suggesting partial lung collapse.",
        "impression": "Atelectasis identified; clinical correlation recommended.",
    },
    "Cardiomegaly": {
        "finding": "Cardiac silhouette is enlarged with a cardiothoracic ratio exceeding 0.5.",
        "impression": "Cardiomegaly present; further cardiac evaluation is recommended.",
    },
    "Effusion": {
        "finding": "Blunting of the costophrenic angle with dependent opacity consistent with pleural effusion.",
        "impression": "Pleural effusion identified; clinical correlation advised.",
    },
    "Infiltration": {
        "finding": "Hazy, ill-defined opacities in the lung parenchyma suggesting pulmonary infiltration.",
        "impression": "Pulmonary infiltration noted; infection or aspiration cannot be excluded.",
    },
    "Mass": {
        "finding": "Focal soft-tissue opacity greater than 3 cm in diameter suspicious for pulmonary mass.",
        "impression": "Pulmonary mass identified; further evaluation with CT and oncology referral recommended.",
    },
    "Nodule": {
        "finding": "Discrete, rounded opacity less than 3 cm in diameter consistent with pulmonary nodule.",
        "impression": "Pulmonary nodule detected; follow-up imaging per Fleischner Society guidelines.",
    },
    "Pneumonia": {
        "finding": "Airspace consolidation with possible air bronchograms in a lobar or segmental distribution consistent with pneumonia.",
        "impression": "Pneumonia is suspected; antibiotic therapy and clinical follow-up are recommended.",
    },
    "Pneumothorax": {
        "finding": "Absent lung markings at the periphery with visible pleural line indicating pneumothorax.",
        "impression": "Pneumothorax identified; urgent clinical assessment required.",
    },
    "Consolidation": {
        "finding": "Homogeneous airspace opacity with air bronchograms indicating lobar or segmental consolidation.",
        "impression": "Pulmonary consolidation present; infectious or inflammatory etiology should be investigated.",
    },
    "Edema": {
        "finding": "Bilateral perihilar haze with Kerley B lines and vascular redistribution consistent with pulmonary edema.",
        "impression": "Pulmonary edema identified; cardiac function should be assessed.",
    },
    "Emphysema": {
        "finding": "Bilateral hyperinflation with flattened hemidiaphragms and increased anteroposterior diameter consistent with emphysema.",
        "impression": "Emphysematous changes noted; pulmonary function testing is recommended.",
    },
    "Fibrosis": {
        "finding": "Reticular interstitial pattern with possible honeycombing in the lower lung zones consistent with pulmonary fibrosis.",
        "impression": "Findings suggest pulmonary fibrosis; high-resolution CT and pulmonology referral recommended.",
    },
    "Pleural_Thickening": {
        "finding": "Irregular pleural thickening along the chest wall without free pleural fluid.",
        "impression": "Pleural thickening identified; prior asbestos exposure or chronic disease should be considered.",
    },
    "Hernia": {
        "finding": "Bowel gas or soft tissue visible above the hemidiaphragm consistent with diaphragmatic hernia.",
        "impression": "Diaphragmatic hernia suspected; clinical and surgical evaluation recommended.",
    },
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
