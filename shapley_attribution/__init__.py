"""
Shapley Attribution: Multi-touch attribution modeling using Shapley values.

A scikit-learn compatible library implementing Shapley value methods for
attribution modeling in online advertising, based on Zhao et al. (2018).

Provides exact and approximate Shapley attribution models, heuristic
baselines, synthetic data generation, and evaluation metrics.
"""

from shapley_attribution.base import BaseAttributionModel
from shapley_attribution.models import (
    SimplifiedShapleyAttribution,
    OrderedShapleyAttribution,
    MonteCarloShapleyAttribution,
)
from shapley_attribution.baselines import (
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
)
from shapley_attribution.datasets import make_attribution_problem
from shapley_attribution.metrics import (
    normalized_mean_absolute_error,
    rank_correlation,
    top_k_overlap,
    attribution_summary,
)

__version__ = "2.0.0"

__all__ = [
    "BaseAttributionModel",
    "SimplifiedShapleyAttribution",
    "OrderedShapleyAttribution",
    "MonteCarloShapleyAttribution",
    "FirstTouchAttribution",
    "LastTouchAttribution",
    "LinearAttribution",
    "TimeDecayAttribution",
    "PositionBasedAttribution",
    "make_attribution_problem",
    "normalized_mean_absolute_error",
    "rank_correlation",
    "top_k_overlap",
    "attribution_summary",
]
