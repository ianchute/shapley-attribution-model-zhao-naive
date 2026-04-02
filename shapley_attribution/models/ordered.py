"""
Ordered (position-aware) Shapley attribution.

Implements the ordered Shapley value from Zhao et al. (2018) where the
position of a channel within a journey affects its attribution score.

Complexity
----------
O(n_journeys * n_touchpoints * 2^n_channels)
Practical for up to ~15 channels.
"""

from itertools import chain, combinations

import numpy as np
from tqdm import tqdm

from shapley_attribution.base import BaseAttributionModel


class OrderedShapleyAttribution(BaseAttributionModel):
    """Exact position-aware Shapley attribution (Zhao et al., 2018).

    Assigns different attribution scores depending on where in the journey
    a channel appears.  More expensive than :class:`SimplifiedShapleyAttribution`
    but captures positional effects.

    Parameters
    ----------
    normalize : bool, default=True
        If True, attribution scores are normalized to sum to 1.
    verbose : bool, default=False
        If True, print progress information during computation.

    Attributes
    ----------
    position_attribution_ : dict
        ``{channel: [score_at_pos_1, score_at_pos_2, ...]}`` after fit.

    References
    ----------
    Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). Shapley Value
    Methods for Attribution Modeling in Online Advertising. arXiv:1804.05327.
    """

    def __init__(self, normalize=True, verbose=False):
        super().__init__(normalize=normalize)
        self.verbose = verbose

    def _compute_attribution(self, X):
        channels = sorted({ch for j in X for ch in j})
        max_touchpoints = max(len(j) for j in X)

        # Pre-index journeys by their unique channel set size
        indexed = {}
        for journey in X:
            s = frozenset(journey)
            size = len(s)
            indexed.setdefault(size, []).append((journey, s))

        # Generate power set
        all_subsets = [
            frozenset(s)
            for s in chain.from_iterable(
                combinations(channels, r) for r in range(1, len(channels) + 1)
            )
        ]

        self.position_attribution_ = {}
        attribution = {}

        desc_iter = tqdm(channels, desc="Ordered Shapley") if self.verbose else channels
        for channel in desc_iter:
            pos_scores = []
            for t in range(1, max_touchpoints + 1):
                score = 0.0
                for subset in all_subsets:
                    if channel not in subset:
                        continue
                    size = len(subset)
                    for journey, journey_set in indexed.get(size, []):
                        if journey_set != subset:
                            continue
                        if t <= len(journey) and journey[t - 1] == channel:
                            score += 1.0 / (journey.count(channel) * size)
                pos_scores.append(score)

            self.position_attribution_[channel] = pos_scores
            attribution[channel] = sum(pos_scores)

        return attribution
