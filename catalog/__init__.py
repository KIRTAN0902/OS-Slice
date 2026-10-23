"""
OS-Slice Model Catalog Package
"""

from .manager import SUPPORTED_MODELS, ModelCatalogManager
from .sliced_models import SLICED_MODELS, SlicedModelManager

__all__ = [
    "SUPPORTED_MODELS",
    "ModelCatalogManager",
    "SLICED_MODELS",
    "SlicedModelManager",
]
