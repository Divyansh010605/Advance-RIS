"""
Unified Models Module for NIH ChestX-ray14 Multi-Disease AI Microservice.
Exports DenseNet-121, ResNet-50, Swin Transformer backbones, ensemble, and ONNX engine.
"""
from models.densenet import (
    NIHDenseNet,
    MultiDiseaseModel,
    TBModel,
    build_densenet121,
    NIH_LABELS,
    _to_xrv_input,
)
from models.resnet import (
    NIHResNet50,
    build_resnet50,
    XRV_RESNET_WEIGHTS,
)
from models.swin import (
    NIHSwin,
    build_swin,
    build_swin_tiny,
    _prepare_swin_input,
)
from models.ensemble import (
    MultiModelEnsemble,
    TestTimeAugmentation,
)
from models.checkpoint import (
    load_checkpoint,
    remap_state_dict_keys,
)
from models.onnx_engine import (
    export_model_to_onnx,
    ONNXInferenceEngine,
)

__all__ = [
    "NIHDenseNet",
    "MultiDiseaseModel",
    "TBModel",
    "build_densenet121",
    "NIH_LABELS",
    "_to_xrv_input",
    "NIHResNet50",
    "build_resnet50",
    "XRV_RESNET_WEIGHTS",
    "NIHSwin",
    "build_swin",
    "build_swin_tiny",
    "_prepare_swin_input",
    "MultiModelEnsemble",
    "TestTimeAugmentation",
    "load_checkpoint",
    "remap_state_dict_keys",
    "export_model_to_onnx",
    "ONNXInferenceEngine",
]
