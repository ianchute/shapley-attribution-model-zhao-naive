"""
Monte Carlo approximation of Shapley values for attribution.

Instead of enumerating all 2^n coalitions, this samples random permutations
and computes marginal contributions.  Convergence is O(1/sqrt(n_iter)) and
the method scales to hundreds of channels.

This implements the ApproShapley algorithm (Castro et al., 2009) adapted for
the multi-touch attribution setting, following the approach used in the SHAP
framework (Lundberg & Lee, 2017).

References
----------
Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the
    Shapley value based on sampling. Computers & Operations Research.
Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model
    Predictions. NeurIPS 2017.
"""

import numpy as np
from tqdm import tqdm

from shapley_attribution.base import BaseAttributionModel


def _conversion_rate(journeys, channel_subset):
    """Compute conversion rate for journeys whose channel set is a
    subset of ``channel_subset``.

    A journey "matches" a coalition if its channel set is a subset of the
    coalition.  The conversion rate is the fraction of matching journeys
    out of all journeys.
    """
    if len(channel_subset) == 0:
        return 0.0
    subset_frozen = frozenset(channel_subset)
    matches = sum(1 for j in journeys if frozenset(j).issubset(subset_frozen))
    return matches / len(journeys) if len(journeys) > 0 else 0.0


class MonteCarloShapleyAttribution(BaseAttributionModel):
    """Approximate Shapley attribution via Monte Carlo permutation sampling.

    Scales to large channel spaces (hundreds of channels) by replacing
    exhaustive subset enumeration with random permutation sampling.

    The marginal contribution of channel *c* in a random permutation *pi*
    is: ``v(S_pi^c ∪ {c}) - v(S_pi^c)`` where ``S_pi^c`` is the set of
    channels preceding *c* in the permutation and ``v`` is a coalition
    value function (here: conversion rate of journeys whose channels are
    within the coalition).

    Parameters
    ----------
    n_iter : int, default=1000
        Number of random permutations to sample.
    random_state : int or None, default=None
        Seed for reproducibility.
    normalize : bool, default=True
        If True, attribution scores are normalized to sum to 1.
    verbose : bool, default=False
        If True, display a progress bar.

    Examples
    --------
    >>> from shapley_attribution import MonteCarloShapleyAttribution
    >>> model = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
    >>> journeys = [[1, 2, 3], [1, 2], [2, 3], [1]]
    >>> model.fit(journeys)
    MonteCarloShapleyAttribution(n_iter=500, random_state=42)
    >>> scores = model.get_attribution()
    """

    def __init__(self, n_iter=1000, random_state=None, normalize=True, verbose=False):
        super().__init__(normalize=normalize)
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def _compute_attribution(self, X):
        rng = np.random.RandomState(self.random_state)
        channels = sorted({ch for j in X for ch in j})
        n_channels = len(channels)
        channel_arr = np.array(channels)

        # Pre-compute journey sets for fast lookup
        journey_sets = [frozenset(j) for j in X]
        n_journeys = len(X)

        # Build a lookup: for each subset (as frozenset) → number of
        # journeys whose channel set is a subset.  We cache results.
        _cache = {}

        def coalition_value(coalition_frozen):
            if coalition_frozen in _cache:
                return _cache[coalition_frozen]
            if len(coalition_frozen) == 0:
                val = 0.0
            else:
                matches = sum(
                    1 for js in journey_sets if js.issubset(coalition_frozen)
                )
                val = matches / n_journeys
            _cache[coalition_frozen] = val
            return val

        shapley = {ch: 0.0 for ch in channels}

        iterator = range(self.n_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="MC Shapley")

        for _ in iterator:
            perm = rng.permutation(n_channels)
            coalition = set()
            prev_value = 0.0
            for idx in perm:
                ch = channel_arr[idx]
                coalition.add(ch)
                curr_value = coalition_value(frozenset(coalition))
                marginal = curr_value - prev_value
                shapley[ch] += marginal
                prev_value = curr_value

        # Average over iterations
        for ch in channels:
            shapley[ch] /= self.n_iter

        # Scale to total conversions
        total = n_journeys
        for ch in channels:
            shapley[ch] *= total

        return shapley
