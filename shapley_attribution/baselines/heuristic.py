"""
Heuristic (rule-based) attribution baselines.

These are standard industry baselines used for comparison with
Shapley-based methods.  They apply deterministic rules to each
*converting* journey.  Non-converting journeys receive no credit.
"""

import math
from collections import defaultdict

import numpy as np

from shapley_attribution.base import BaseAttributionModel


class FirstTouchAttribution(BaseAttributionModel):
    """Assign 100% credit to the first channel in each converting journey.

    Parameters
    ----------
    normalize : bool, default=True
        If True, aggregate scores are normalized to sum to 1.
    """

    def _compute_attribution(self, X):
        scores = defaultdict(float)
        for i, journey in enumerate(X):
            if self.conversions_[i]:
                scores[journey[0]] += 1.0
        return dict(scores)

    def _attribute_single(self, journey):
        return {journey[0]: 1.0}


class LastTouchAttribution(BaseAttributionModel):
    """Assign 100% credit to the last channel in each converting journey.

    Parameters
    ----------
    normalize : bool, default=True
        If True, aggregate scores are normalized to sum to 1.
    """

    def _compute_attribution(self, X):
        scores = defaultdict(float)
        for i, journey in enumerate(X):
            if self.conversions_[i]:
                scores[journey[-1]] += 1.0
        return dict(scores)

    def _attribute_single(self, journey):
        return {journey[-1]: 1.0}


class LinearAttribution(BaseAttributionModel):
    """Distribute credit equally across all touchpoints in converting journeys.

    Parameters
    ----------
    normalize : bool, default=True
        If True, aggregate scores are normalized to sum to 1.
    """

    def _compute_attribution(self, X):
        scores = defaultdict(float)
        for i, journey in enumerate(X):
            if self.conversions_[i]:
                credit = 1.0 / len(journey)
                for ch in journey:
                    scores[ch] += credit
        return dict(scores)

    def _attribute_single(self, journey):
        credit = 1.0 / len(journey)
        result = defaultdict(float)
        for ch in journey:
            result[ch] += credit
        return dict(result)


class TimeDecayAttribution(BaseAttributionModel):
    """Give more credit to channels closer to conversion (end of journey).

    Uses exponential decay: weight at position *i* (0-indexed from start)
    is ``decay_rate ^ (n - 1 - i)`` where *n* is the journey length.

    Parameters
    ----------
    decay_rate : float, default=0.5
        Base of the exponential decay.  Smaller values give the last
        touchpoint more relative credit.
    normalize : bool, default=True
        If True, aggregate scores are normalized to sum to 1.
    """

    def __init__(self, decay_rate=0.5, normalize=True):
        super().__init__(normalize=normalize)
        self.decay_rate = decay_rate

    def _compute_attribution(self, X):
        scores = defaultdict(float)
        for i, journey in enumerate(X):
            if self.conversions_[i]:
                weights = self._journey_weights(journey)
                total_w = sum(weights)
                for ch, w in zip(journey, weights):
                    scores[ch] += w / total_w if total_w > 0 else 0
        return dict(scores)

    def _attribute_single(self, journey):
        weights = self._journey_weights(journey)
        total_w = sum(weights)
        result = defaultdict(float)
        for ch, w in zip(journey, weights):
            result[ch] += w / total_w if total_w > 0 else 0
        return dict(result)

    def _journey_weights(self, journey):
        n = len(journey)
        return [self.decay_rate ** (n - 1 - i) for i in range(n)]


class PositionBasedAttribution(BaseAttributionModel):
    """Give 40% credit to first touch, 40% to last touch, 20% split among middle.

    Also known as the "U-shaped" or "bathtub" model.

    Parameters
    ----------
    first_weight : float, default=0.4
        Credit allocated to the first touchpoint.
    last_weight : float, default=0.4
        Credit allocated to the last touchpoint.
    normalize : bool, default=True
        If True, aggregate scores are normalized to sum to 1.
    """

    def __init__(self, first_weight=0.4, last_weight=0.4, normalize=True):
        super().__init__(normalize=normalize)
        self.first_weight = first_weight
        self.last_weight = last_weight

    def _compute_attribution(self, X):
        scores = defaultdict(float)
        for i, journey in enumerate(X):
            if self.conversions_[i]:
                weights = self._journey_weights(journey)
                for ch, w in zip(journey, weights):
                    scores[ch] += w
        return dict(scores)

    def _attribute_single(self, journey):
        weights = self._journey_weights(journey)
        result = defaultdict(float)
        for ch, w in zip(journey, weights):
            result[ch] += w
        return dict(result)

    def _journey_weights(self, journey):
        n = len(journey)
        if n == 1:
            return [1.0]
        if n == 2:
            return [self.first_weight + (1 - self.first_weight - self.last_weight) / 2,
                    self.last_weight + (1 - self.first_weight - self.last_weight) / 2]
        middle_weight = (1.0 - self.first_weight - self.last_weight) / (n - 2)
        return [self.first_weight] + [middle_weight] * (n - 2) + [self.last_weight]
