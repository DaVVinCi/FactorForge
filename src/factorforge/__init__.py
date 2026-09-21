"""FactorForge: a small, reproducible factor-research pipeline."""

from .data import generate_demo_data
from .dsl import ExpressionEngine, ExpressionValidationError
from .pipeline import DemoPipeline

__all__ = [
    "DemoPipeline",
    "ExpressionEngine",
    "ExpressionValidationError",
    "generate_demo_data",
]

__version__ = "0.1.0"

