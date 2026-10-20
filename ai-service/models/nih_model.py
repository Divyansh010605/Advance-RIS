"""
Unified Model Manager for NIH ChestX-ray14 Multi-Disease Chest X-Ray Screening.
Aliases TBModel to NIHModel for clean, modern naming.
"""
from models.tb_model import TBModel, MultiDiseaseModel

NIHModel = TBModel

__all__ = ["NIHModel", "TBModel", "MultiDiseaseModel"]
