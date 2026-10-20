"""
Pathology Consistency & Hallucination Checker.
Ensures generated radiology reports do not make unsupported pathology claims.
"""
from typing import List, Dict, Tuple, Set

# Aliases used to detect positive claims about each NIH class in free-form report text.
# Negative phrases ("no effusion", "without atelectasis") are already handled by the checker.
DISEASE_ALIASES: Dict[str, Set[str]] = {
    "Atelectasis":         {"atelectasis", "plate-like opacity", "subsegmental collapse", "linear opacity"},
    "Cardiomegaly":        {"cardiomegaly", "enlarged heart", "cardiac enlargement", "increased cardiac silhouette"},
    "Effusion":            {"effusion", "pleural effusion", "costophrenic blunting", "pleural fluid"},
    "Infiltration":        {"infiltration", "pulmonary infiltrate", "hazy opacity", "infiltrate"},
    "Mass":                {"mass", "pulmonary mass", "lung mass", "soft-tissue opacity"},
    "Nodule":              {"nodule", "pulmonary nodule", "lung nodule", "rounded opacity"},
    "Pneumonia":           {"pneumonia", "airspace consolidation", "lobar consolidation", "air bronchogram"},
    "Pneumothorax":        {"pneumothorax", "pleural line", "absent lung markings", "collapsed lung"},
    "Consolidation":       {"consolidation", "lobar opacity", "airspace disease", "homogeneous opacity"},
    "Edema":               {"edema", "pulmonary edema", "perihilar haze", "kerley b", "vascular redistribution"},
    "Emphysema":           {"emphysema", "hyperinflation", "flattened diaphragm", "barrel chest"},
    "Fibrosis":            {"fibrosis", "pulmonary fibrosis", "reticular pattern", "honeycombing"},
    "Pleural_Thickening":  {"pleural thickening", "pleural_thickening", "thickened pleura", "pleural scarring"},
    "Hernia":              {"hernia", "diaphragmatic hernia", "bowel above diaphragm", "hiatal hernia"},
}



class ReportGroundingChecker:
    def __init__(self, target_classes: List[str]):
        self.target_classes = target_classes

    def validate_report(self, detected_diseases: List[str], generated_text: str) -> Tuple[bool, List[str]]:
        """
        Validates that generated text mentions only pathologies present in detected_diseases.
        
        Returns:
            (is_grounded: bool, unsupported_findings: List[str])
        """
        text_lower = generated_text.lower()
        unsupported = []
        
        detected_set = set(detected_diseases)

        for disease, aliases in DISEASE_ALIASES.items():
            if disease in detected_set:
                continue
                
            # Check if an undetected disease is mentioned in the text as a positive finding
            for alias in aliases:
                # Basic negative check (e.g., "no effusion", "without pneumothorax")
                negative_phrases = [f"no {alias}", f"without {alias}", f"absence of {alias}", f"negative for {alias}", f"clear of {alias}"]
                if alias in text_lower and not any(neg in text_lower for neg in negative_phrases):
                    unsupported.append(disease)
                    break

        unsupported_unique = list(set(unsupported))
        is_grounded = len(unsupported_unique) == 0
        return is_grounded, unsupported_unique
