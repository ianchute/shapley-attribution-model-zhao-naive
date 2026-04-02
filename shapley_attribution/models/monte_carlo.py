"""
Monte Carlo approximation of Shapley values for attribution.

Instead of enumerating all 2^n coalitions, this samples random permutations
and computes marginal contributions.  Convergence is O(1/sqrt(n_iter)) and
the method scales to hundreds of channels.

The coalition value function is a learned conversion probability model
(GBM by default).  Coalition values are computed using interventional
Shapley: v(S) = P(conversion | exactly channels S present).  This is a
deterministic function of the coalition mask with no background averaging,
eliminating a major source of variance.

References
----------
Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the
    Shapley value based on sampling. Computers & Operations Research.
Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model
    Predictions. NeurIPS 2017.
Janzing, D., Minorics, L., & Bloebaum, P. (2020). Feature relevance
    quantification in explainable AI: A causal problem. AISTATS 2020.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

from shapley_attribution.base import BaseAttributionModel


class MonteCarloShapleyAttribution(BaseAttributionModel):
    """Approximate Shapley attribution via Monte Carlo permutation sampling.

    Scales to large channel spaces (hundreds of channels) by replacing
    exhaustive subset enumeration with random permutation sampling.

    Uses a learned value function: a GBM classifier is trained on
    (binary_presence, converted) pairs.  The coalition value v(S) is
    the model's predicted conversion probability for the exact binary
    mask where only channels in S are present (interventional approach).

    Parameters
    ----------
    n_iter : int, default=2000
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
    >>> model = MonteCarloShapleyAttribution(n_iter=1000, random_state=42)
    >>> journeys = [[1, 2, 3], [1, 2], [2, 3], [1]]
    >>> conversions = [1, 1, 0, 0]
    >>> model.fit(journeys, conversions)
    MonteCarloShapleyAttribution(n_iter=1000, random_state=42)
    >>> scores = model.get_attribution()
    """

    def __init__(
        self,
        n_iter=2000,
        random_state=None,
        normalize=True,
        verbose=False,
    ):
        super().__init__(normalize=normalize)
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the attribution model.

        Parameters
        ----------
        X : list of list of int/str
            Customer journeys.
        y : array-like of shape (n_journeys,) or None
            Binary conversion labels (1=converted, 0=not).
            If None, all journeys are assumed to be converting (legacy mode).

        Returns
        -------
        self
        """
        X = self._validate_journeys(X)
        self.channels_ = np.array(sorted({ch for journey in X for ch in journey}))
        self.channel_to_idx_ = {ch: i for i, ch in enumerate(self.channels_)}

        if y is not None:
            y = np.asarray(y, dtype=int)
            if len(y) != len(X):
                raise ValueError(
                    f"Length mismatch: {len(X)} journeys but {len(y)} labels."
                )
            self.conversions_ = y
        else:
            self.conversions_ = np.ones(len(X), dtype=int)

        self.attribution_ = self._compute_attribution(X)
        self.is_fitted_ = True
        return self

    def _journeys_to_features(self, X):
        """Convert journeys to a binary presence feature matrix."""
        n_channels = len(self.channels_)
        features = np.zeros((len(X), n_channels))
        for i, journey in enumerate(X):
            for ch in set(journey):
                if ch in self.channel_to_idx_:
                    features[i, self.channel_to_idx_[ch]] = 1.0
        return features

    def _compute_attribution(self, X):
        rng = np.random.RandomState(self.random_state)
        n_channels = len(self.channels_)

        # ---- Train conversion model on binary presence features ----
        features = self._journeys_to_features(X)
        y = self.conversions_

        has_both_classes = len(np.unique(y)) >= 2
        if has_both_classes:
            n_samples = len(y)
            self.value_model_ = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=min(4, max(1, n_samples // 10)),
                learning_rate=0.1,
                random_state=self.random_state,
                subsample=min(0.8, 1.0) if n_samples > 20 else 1.0,
                min_samples_leaf=max(1, min(20, n_samples // 5)),
            )
            self.value_model_.fit(features, y)
        else:
            # Only one class (e.g., all journeys convert) — the learned
            # model would predict a constant, giving zero marginal
            # contributions.  Fall back to set-based Shapley: credit
            # each converting journey equally among its channels.
            self.value_model_ = None
            from collections import Counter
            converting = [j for i, j in enumerate(X) if y[i]]
            journey_sets = Counter(frozenset(j) for j in converting)
            fallback_attr = {}
            for ch in self.channels_:
                score = 0.0
                for jset, count in journey_sets.items():
                    if ch in jset:
                        score += count / len(jset)
                fallback_attr[ch] = score
            return fallback_attr

        # ---- Interventional coalition value function ----
        # v(S) = f(mask_S) where mask_S is the exact binary vector
        # with 1s for channels in S and 0s elsewhere.
        # No background averaging — deterministic and zero-variance
        # for each coalition.
        _cache = {}

        def coalition_value(mask_tuple):
            if mask_tuple in _cache:
                return _cache[mask_tuple]

            if self.value_model_ is None:
                _cache[mask_tuple] = self._fallback_value
                return self._fallback_value

            mask = np.array(mask_tuple, dtype=float).reshape(1, -1)
            val = float(self.value_model_.predict_proba(mask)[0, 1])
            _cache[mask_tuple] = val
            return val

        # ---- Monte Carlo Shapley via permutation sampling ----
        shapley = np.zeros(n_channels)

        iterator = range(self.n_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="MC Shapley")

        for _ in iterator:
            perm = rng.permutation(n_channels)
            active = np.zeros(n_channels)
            prev_value = coalition_value(tuple(active))

            for idx in perm:
                active[idx] = 1.0
                curr_value = coalition_value(tuple(active))
                shapley[idx] += curr_value - prev_value
                prev_value = curr_value

        shapley /= self.n_iter

        # Scale: Shapley values are in probability space [0, 1].
        # Scale to total conversions so they're comparable with baselines.
        n_conversions = max(self.conversions_.sum(), 1)
        shapley *= n_conversions

        # Map back to channel names
        attribution = {}
        for i, ch in enumerate(self.channels_):
            attribution[ch] = max(shapley[i], 0.0)

        return attribution
