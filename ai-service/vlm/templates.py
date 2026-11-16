"""
Structured ACR Radiology Report Templates for NIH ChestX-ray14.
Defines clinically calibrated findings and impressions for thoracic pathologies.
"""
from typing import Dict

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

NORMAL_TEMPLATE = {
    "finding": "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. Cardiomediastinal silhouette and osseous structures appear unremarkable.",
    "impression": "No acute cardiopulmonary abnormality detected.",
}

__all__ = ["TEMPLATES", "NORMAL_TEMPLATE"]
