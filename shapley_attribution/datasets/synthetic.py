"""
Synthetic data generators for attribution model benchmarking.

Generates customer journeys with a known ground-truth conversion model
so that attribution methods can be evaluated against true channel importance.

Key design: all channels appear with roughly equal frequency, but
conversion depends on which channels appear *together*.  This creates
a setting where Shapley-based methods (which measure marginal contribution)
should outperform frequency-based heuristics.
"""

import numpy as np
from scipy.special import expit  # sigmoid


def make_attribution_problem(
    n_channels=6,
    n_journeys=5000,
    max_journey_length=8,
    min_journey_length=2,
    channel_importance=None,
    interaction_effects=0.5,
    noise=0.1,
    base_conversion_rate=0.3,
    random_state=None,
):
    """Generate a synthetic attribution problem with known ground truth.

    Creates customer journeys where conversion probability depends on
    channel presence through a logistic model with known coefficients
    and pairwise interaction terms.

    Crucially, channel *sampling* probabilities are roughly uniform —
    the signal is in which combinations drive conversion, not in which
    channels appear more often.  This rewards models that measure
    marginal contribution (Shapley) over those that count frequency
    (heuristics).

    Parameters
    ----------
    n_channels : int, default=6
        Number of distinct marketing channels.
    n_journeys : int, default=5000
        Total number of customer journeys (converting + non-converting).
    max_journey_length : int, default=8
        Maximum number of touchpoints per journey.
    min_journey_length : int, default=2
        Minimum number of touchpoints per journey.
    channel_importance : array-like of shape (n_channels,) or None
        True importance weight for each channel.  If None, random
        importances are generated.
    interaction_effects : float, default=0.5
        Strength of pairwise synergy.  Higher values make channel
        combinations more important relative to individual presence.
    noise : float, default=0.1
        Noise in channel sampling.
    base_conversion_rate : float, default=0.3
        Target overall conversion rate.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    journeys : list of list of int
        All customer journeys (both converting and non-converting).
    conversions : ndarray of shape (n_journeys,)
        Binary labels: 1 if the journey converted, 0 otherwise.
    ground_truth : ndarray of shape (n_channels,)
        Normalized true attribution weights.
    channel_names : list of int
        Channel identifiers (0 to n_channels-1).
    """
    rng = np.random.RandomState(random_state)
    channel_names = list(range(n_channels))

    # ---- Channel importance (conversion coefficients) ----
    if channel_importance is None:
        raw = rng.dirichlet(np.ones(n_channels) * 2.0)
        raw = raw ** 1.3
        ground_truth = raw / raw.sum()
    else:
        ground_truth = np.array(channel_importance, dtype=float)
        ground_truth = ground_truth / ground_truth.sum()

    # Conversion model coefficients — scaled so they produce real signal
    channel_coefs = ground_truth * n_channels * 3.0

    # ---- Pairwise interaction matrix ----
    interaction_matrix = np.zeros((n_channels, n_channels))
    if interaction_effects > 0:
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                synergy = rng.uniform(-0.3, 1.0) * interaction_effects
                synergy *= (ground_truth[i] + ground_truth[j]) * n_channels
                interaction_matrix[i, j] = synergy
                interaction_matrix[j, i] = synergy

    # ---- Calibrate intercept for target conversion rate ----
    # Monte Carlo estimate: generate some journeys and find the intercept
    # that gives the target conversion rate
    cal_logits = []
    for _ in range(2000):
        length = rng.randint(min_journey_length, max_journey_length + 1)
        journey = rng.choice(channel_names, size=length)
        present = np.zeros(n_channels)
        for ch in set(journey):
            present[ch] = 1.0
        logit_p = np.dot(channel_coefs, present)
        for i in range(n_channels):
            if present[i] == 0:
                continue
            for j in range(i + 1, n_channels):
                if present[j] == 0:
                    continue
                logit_p += interaction_matrix[i, j]
        cal_logits.append(logit_p)

    cal_logits = np.array(cal_logits)
    # Find intercept so that sigmoid(intercept + logits).mean() ≈ target
    from scipy.optimize import brentq

    def _conv_rate(intercept):
        return expit(intercept + cal_logits).mean() - base_conversion_rate

    try:
        intercept = brentq(_conv_rate, -20, 20)
    except ValueError:
        intercept = -np.median(cal_logits)

    # ---- Generate journeys with ~uniform channel sampling ----
    journeys = []
    conversions = []

    for _ in range(n_journeys):
        length = rng.randint(min_journey_length, max_journey_length + 1)

        # Near-uniform sampling — all channels have similar appearance rates
        probs = np.ones(n_channels) + rng.normal(0, noise, n_channels)
        probs = np.clip(probs, 0.3, 1.7)
        probs = probs / probs.sum()

        journey = list(rng.choice(channel_names, size=length, p=probs))
        journeys.append(journey)

        # ---- Conversion probability ----
        present = np.zeros(n_channels)
        for ch in set(journey):
            present[ch] = 1.0

        logit_p = intercept + np.dot(channel_coefs, present)

        for i in range(n_channels):
            if present[i] == 0:
                continue
            for j in range(i + 1, n_channels):
                if present[j] == 0:
                    continue
                logit_p += interaction_matrix[i, j]

        conv_prob = expit(logit_p)
        converted = rng.random() < conv_prob
        conversions.append(int(converted))

    conversions = np.array(conversions)

    return journeys, conversions, ground_truth, channel_names
