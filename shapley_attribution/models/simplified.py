"""
Simplified (set-based, position-agnostic) Shapley attribution.

Implements the set-based Shapley value from Zhao et al. (2018) where each
journey is treated as an unordered set of channels.  Attribution is exact
but has exponential worst-case complexity in the number of unique channels.

Complexity
----------
O(n_journeys * n_channels)  — practical for up to ~20 channels.
"""

from collections import Counter
from itertools import chain, combinations

import numpy as np
from tqdm import tqdm

from shapley_attribution.base import BaseAttributionModel


class SimplifiedShapleyAttribution(BaseAttributionModel):
    """Exact set-based Shapley attribution (Zhao et al., 2018).

    Treats each journey as an unordered set of channels and computes exact
    Shapley values.  Only converting journeys contribute to attribution.

    Parameters
    ----------
    normalize : bool, default=True
        If True, attribution scores are normalized to sum to 1.
    verbose : bool, default=False
        If True, print progress information during computation.

    References
    ----------
    Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). Shapley Value
    Methods for Attribution Modeling in Online Advertising. arXiv:1804.05327.
    """

    def __init__(self, normalize=True, verbose=False):
        super().__init__(normalize=normalize)
        self.verbose = verbose

    def _compute_attribution(self, X):
        # Only count converting journeys
        converting = [
            j for i, j in enumerate(X) if self.conversions_[i]
        ]

        journey_sets = Counter(frozenset(j) for j in converting)
        channels = sorted({ch for j in X for ch in j})

        attribution = {}
        iterable = tqdm(channels, desc="Shapley") if self.verbose else channels
        for channel in iterable:
            score = 0.0
            for journey_set, count in journey_sets.items():
                if channel in journey_set:
                    score += count / len(journey_set)
            attribution[channel] = score

        return attribution

    def _attribute_single(self, journey):
        """Equal credit among channels in the journey (set-based)."""
        unique = set(journey)
        n = len(unique)
        return {ch: 1.0 / n for ch in unique} if n > 0 else {}
