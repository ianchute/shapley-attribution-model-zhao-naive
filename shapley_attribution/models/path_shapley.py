"""
Path Shapley attribution — ordering-aware Shapley using the actual journey
sequence as the coalition permutation rather than averaging over random ones.

For a converting journey [c₁, c₂, ..., cₙ] (deduplicated, preserving order):

    contribution(cᵢ) = v({c₁, …, cᵢ}) − v({c₁, …, cᵢ₋₁})

where v(S) = GBM.predict_proba(binary_mask(S)).

This makes journeys [A, B] and [B, A] produce different attributions:
  [A, B]:  A gets v({A}) − v({}),  B gets v({A,B}) − v({A})
  [B, A]:  B gets v({B}) − v({}),  A gets v({A,B}) − v({B})

The ordering sensitivity flows from which channels form the predecessor
coalition at each step — not from any positional signal inside the GBM.
The value function itself remains set-based (binary mask), so the model
benefits from a directed synthetic DGP (make_attribution_problem with
directed_interaction_strength > 0) that bakes asymmetric synergies into
conversion probability.

Computational complexity: O(n_journeys × max_journey_length) GBM calls,
all deduplicated and cached.  Typically faster than MC Shapley for the
same dataset.

References
----------
Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). Shapley Value Methods
    for Attribution Modeling in Online Advertising. arXiv:1804.05327.
    [Ordered Shapley formulation, Section 3.2]
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from shapley_attribution.base import BaseAttributionModel


class PathShapleyAttribution(BaseAttributionModel):
    """Ordering-aware Shapley attribution via actual journey path permutations.

    Trains a GradientBoostingClassifier on (binary_presence, converted) pairs
    identically to :class:`MonteCarloShapleyAttribution`, then evaluates each
    converting journey's actual channel sequence as the coalition-formation
    order.  The marginal contribution of each channel is computed against the
    prefix of channels that appeared before it.

    Unlike MC Shapley, this model is *not* symmetric in channels — reversing
    a journey changes which channel is credited for which synergy.

    Parameters
    ----------
    normalize : bool, default=True
        If True, attribution scores sum to 1.
    random_state : int or None, default=None
        Seed for the GBM.
    verbose : bool, default=False

    Attributes
    ----------
    value_model_ : GradientBoostingClassifier or None
        Trained conversion model (None if only one class in training data).
    position_attribution_ : dict[channel, list[float]]
        ``{ch: [score_at_pos_0, score_at_pos_1, ...]}``  Per-position credit
        breakdown across all converting journeys.  Useful for visualising
        upper-funnel vs lower-funnel behaviour.
    """

    def __init__(self, normalize=True, random_state=None, verbose=False):
        super().__init__(normalize=normalize)
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the model.

        Parameters
        ----------
        X : list of list
            Customer journeys (ordered).
        y : array-like of shape (n_journeys,) or None
            Binary conversion labels.  If None all journeys are assumed
            converting (legacy mode).

        Returns
        -------
        self
        """
        X = self._validate_journeys(X)
        self.channels_ = np.array(sorted({ch for journey in X for ch in journey}))
        self.channel_to_idx_ = {ch: i for i, ch in enumerate(self.channels_)}

        if y is not None:
            y_arr = np.asarray(y, dtype=int)
            if len(y_arr) != len(X):
                raise ValueError(
                    f"Length mismatch: {len(X)} journeys but {len(y_arr)} labels."
                )
            self.conversions_ = y_arr
        else:
            self.conversions_ = np.ones(len(X), dtype=int)

        self.attribution_ = self._compute_attribution(X)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _journeys_to_features(self, X):
        n_channels = len(self.channels_)
        features = np.zeros((len(X), n_channels))
        for i, journey in enumerate(X):
            for ch in set(journey):
                if ch in self.channel_to_idx_:
                    features[i, self.channel_to_idx_[ch]] = 1.0
        return features

    def _train_value_model(self, X, y):
        """Train GBM or fall back to set-based Shapley for single-class data."""
        features = self._journeys_to_features(X)
        has_both_classes = len(np.unique(y)) >= 2

        if has_both_classes:
            n_samples = len(y)
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=min(4, max(1, n_samples // 10)),
                learning_rate=0.1,
                random_state=self.random_state,
                subsample=0.8 if n_samples > 20 else 1.0,
                min_samples_leaf=max(1, min(20, n_samples // 5)),
            )
            model.fit(features, y)
            return model
        return None  # triggers fallback

    def _make_coalition_value_fn(self, n_channels):
        """Return a cached coalition-value callable."""
        _cache = {}
        fallback_val = float(self.conversions_.mean())

        def coalition_value(mask_tuple):
            if mask_tuple in _cache:
                return _cache[mask_tuple]
            if self.value_model_ is None:
                _cache[mask_tuple] = fallback_val
                return fallback_val
            mask = np.array(mask_tuple, dtype=float).reshape(1, -1)
            val = float(self.value_model_.predict_proba(mask)[0, 1])
            _cache[mask_tuple] = val
            return val

        return coalition_value

    def _compute_attribution(self, X):
        n_channels = len(self.channels_)
        y = self.conversions_

        # ---- Train value model ----
        self.value_model_ = self._train_value_model(X, y)

        # Single-class fallback: equal Shapley among channels in journey
        if self.value_model_ is None:
            from collections import Counter
            converting = [j for j, c in zip(X, y) if c]
            journey_sets = Counter(frozenset(j) for j in converting)
            fallback = {}
            for ch in self.channels_:
                score = 0.0
                for jset, count in journey_sets.items():
                    if ch in jset:
                        score += count / len(jset)
                fallback[ch] = score
            # Empty position_attribution_ for API consistency
            self.position_attribution_ = {ch: [] for ch in self.channels_}
            return fallback

        coalition_value = self._make_coalition_value_fn(n_channels)

        # v({}) — baseline with no channels present
        zero_mask = tuple(np.zeros(n_channels))
        v_empty = coalition_value(zero_mask)

        # ---- Accumulate path-marginal contributions ----
        channel_scores = {ch: 0.0 for ch in self.channels_}
        max_pos = max(len(j) for j in X)
        position_attribution = {ch: [0.0] * max_pos for ch in self.channels_}

        iterator = zip(X, y)
        if self.verbose:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Path Shapley")

        for journey, converted in iterator:
            if not converted:
                continue

            # Deduplicate preserving first-occurrence order
            seen = set()
            unique_seq = []
            for ch in journey:
                if ch in self.channel_to_idx_ and ch not in seen:
                    seen.add(ch)
                    unique_seq.append(ch)

            if not unique_seq:
                continue

            mask = list(np.zeros(n_channels))
            prev_value = v_empty

            for position, ch in enumerate(unique_seq):
                mask[self.channel_to_idx_[ch]] = 1.0
                current_value = coalition_value(tuple(mask))
                contribution = current_value - prev_value

                channel_scores[ch] += max(contribution, 0.0)
                position_attribution[ch][position] += max(contribution, 0.0)
                prev_value = current_value

        self.position_attribution_ = position_attribution

        # Normalize
        total = sum(channel_scores.values())
        if total > 0 and self.normalize:
            channel_scores = {ch: v / total for ch, v in channel_scores.items()}

        return channel_scores

    def _attribute_single(self, journey):
        """Per-journey path attribution for transform()."""
        self._check_is_fitted()
        n_channels = len(self.channels_)
        coalition_value = self._make_coalition_value_fn(n_channels)
        zero_mask = tuple(np.zeros(n_channels))
        v_empty = coalition_value(zero_mask)

        seen = set()
        unique_seq = []
        for ch in journey:
            if ch in self.channel_to_idx_ and ch not in seen:
                seen.add(ch)
                unique_seq.append(ch)

        scores = {}
        mask = list(np.zeros(n_channels))
        prev_value = v_empty

        for ch in unique_seq:
            mask[self.channel_to_idx_[ch]] = 1.0
            current_value = coalition_value(tuple(mask))
            scores[ch] = max(current_value - prev_value, 0.0)
            prev_value = current_value

        total = sum(scores.values())
        if self.normalize and total > 0:
            scores = {ch: v / total for ch, v in scores.items()}

        return scores

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_position_attribution(self, ax=None, top_k=None, title=None):
        """Stacked bar chart: per-channel credit broken down by journey position.

        Shows whether a channel earns credit early (upper-funnel) or late
        (lower-funnel / closer to conversion).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
        top_k : int or None — show only top-k channels by total score
        title : str or None

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from shapley_attribution.visualization import plot_position_attribution
        return plot_position_attribution(self, ax=ax, top_k=top_k, title=title)
