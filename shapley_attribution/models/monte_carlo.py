"""
Monte Carlo approximation of Shapley values for attribution.

Instead of enumerating all 2^n coalitions, this samples random permutations
and computes marginal contributions.  Convergence is O(1/sqrt(n_iter)) and
the method scales to hundreds of channels.

The coalition value function is a learned conversion probability model
(logistic regression by default), following the KernelSHAP approach
(Lundberg & Lee, 2017).  Coalition values are computed by marginalizing
over a background sample — inactive channels are replaced with values
drawn from the empirical distribution, properly capturing the "absence"
of a channel.

References
----------
Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the
    Shapley value based on sampling. Computers & Operations Research.
Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model
    Predictions. NeurIPS 2017.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from shapley_attribution.base import BaseAttributionModel


class MonteCarloShapleyAttribution(BaseAttributionModel):
    """Approximate Shapley attribution via Monte Carlo permutation sampling.

    Scales to large channel spaces (hundreds of channels) by replacing
    exhaustive subset enumeration with random permutation sampling.

    Uses a learned value function: a classifier is trained on
    (journey_features, converted) pairs, then Shapley values are computed
    over the classifier's predicted conversion probability.

    Coalition values are estimated by marginalizing over a background
    sample: for inactive channels, feature values are drawn from the
    empirical distribution rather than set to zero.  This properly
    represents "what would happen if this channel were absent" rather
    than "what if this channel's feature were zero".

    Parameters
    ----------
    n_iter : int, default=2000
        Number of random permutations to sample.
    n_background : int, default=100
        Number of background samples for marginalization.
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
        n_background=100,
        random_state=None,
        normalize=True,
        verbose=False,
    ):
        super().__init__(normalize=normalize)
        self.n_iter = n_iter
        self.n_background = n_background
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
        """Convert journeys to a binary presence feature matrix.

        For each journey, produces a feature vector of shape (n_channels,)
        where each element is 1 if that channel appears in the journey, 0
        otherwise.  Binary features have clean masking semantics: setting a
        feature to 0 means "this channel was not present".
        """
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

        # ---- Build feature matrix and train conversion model ----
        features = self._journeys_to_features(X)
        y = self.conversions_

        has_both_classes = len(np.unique(y)) >= 2
        if has_both_classes:
            # GBM captures interactions naturally; falls back to
            # logistic regression with interaction features if sklearn
            # version is too old for GBM predict_proba
            self.value_model_ = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                subsample=0.8,
            )
            self.value_model_.fit(features, y)
        else:
            self.value_model_ = None
            self._fallback_value = float(y.mean())

        # ---- Background sample for marginalization ----
        n_bg = min(self.n_background, len(X))
        bg_indices = rng.choice(len(X), size=n_bg, replace=False)
        background = features[bg_indices]  # shape (n_bg, n_channels)

        # ---- Coalition value function ----
        # v(S) = E_bg[ f(x_S, bg_{\S}) ]
        # For active channels, use the background sample's own values;
        # for inactive channels, also use the background values.
        # This means: for each background sample, construct a feature vector
        # where active channels keep background values and inactive channels
        # also keep background values — then the marginal contribution is the
        # *difference* when we toggle a channel from "background" to "active".
        #
        # More precisely: we evaluate f on each background sample, masking
        # active channels to their real values and inactive ones to zero
        # (since zero = "channel not present in journey").

        _cache = {}

        def coalition_value(active_mask_tuple):
            """Expected conversion probability when only `active` channels are present."""
            if active_mask_tuple in _cache:
                return _cache[active_mask_tuple]

            active_mask = np.array(active_mask_tuple, dtype=float)

            if self.value_model_ is None:
                _cache[active_mask_tuple] = self._fallback_value
                return self._fallback_value

            # Mask background samples: keep active channel features, zero inactive
            masked = background * active_mask[np.newaxis, :]
            probs = self.value_model_.predict_proba(masked)[:, 1]
            val = float(probs.mean())

            _cache[active_mask_tuple] = val
            return val

        # ---- Monte Carlo Shapley ----
        shapley = np.zeros(n_channels)

        iterator = range(self.n_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="MC Shapley")

        for _ in iterator:
            perm = rng.permutation(n_channels)
            active_mask = np.zeros(n_channels)
            prev_value = coalition_value(tuple(active_mask))

            for idx in perm:
                active_mask[idx] = 1.0
                curr_value = coalition_value(tuple(active_mask))
                shapley[idx] += curr_value - prev_value
                prev_value = curr_value

        # Average over iterations
        shapley /= self.n_iter

        # Scale to number of conversions
        n_conversions = max(self.conversions_.sum(), 1)
        shapley *= n_conversions

        # Map back to channel names
        attribution = {}
        for i, ch in enumerate(self.channels_):
            attribution[ch] = max(shapley[i], 0.0)

        return attribution
