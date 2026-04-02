"""
Synthetic data generators for attribution model benchmarking.

Generates customer journeys from a known ground-truth attribution model
so that methods can be evaluated against the true channel importance.
"""

import numpy as np


def make_attribution_problem(
    n_channels=6,
    n_journeys=5000,
    max_journey_length=8,
    min_journey_length=1,
    channel_importance=None,
    interaction_effects=None,
    noise=0.1,
    random_state=None,
):
    """Generate a synthetic attribution problem with known ground truth.

    Creates customer journeys where the probability of conversion and the
    "true" channel importances are controlled.  Channels with higher
    importance appear more often in converting journeys.

    Parameters
    ----------
    n_channels : int, default=6
        Number of distinct marketing channels.
    n_journeys : int, default=5000
        Number of customer journeys to generate.
    max_journey_length : int, default=8
        Maximum number of touchpoints per journey.
    min_journey_length : int, default=1
        Minimum number of touchpoints per journey.
    channel_importance : array-like of shape (n_channels,) or None
        True importance weight for each channel.  If None, random
        importances are generated from a Dirichlet distribution.
    interaction_effects : float or None, default=None
        If not None, adds pairwise synergy between channel pairs.
        Value between 0 and 1 controls the strength of interactions.
    noise : float, default=0.1
        Amount of noise added to the generative process.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    journeys : list of list of int
        Simulated customer journeys (converting only).
    ground_truth : ndarray of shape (n_channels,)
        Normalized true attribution weights.  These are the generative
        importances, providing a ground truth for benchmarking.
    channel_names : list of int
        Channel identifiers (0 to n_channels-1).

    Examples
    --------
    >>> from shapley_attribution.datasets import make_attribution_problem
    >>> journeys, truth, channels = make_attribution_problem(
    ...     n_channels=5, n_journeys=2000, random_state=42
    ... )
    >>> len(journeys)
    2000
    >>> truth.sum()  # doctest: +SKIP
    1.0
    """
    rng = np.random.RandomState(random_state)
    channel_names = list(range(n_channels))

    # Generate true importance weights
    if channel_importance is None:
        raw = rng.dirichlet(np.ones(n_channels) * 2)
        # Make it more skewed so differences are detectable
        raw = raw ** 1.5
        ground_truth = raw / raw.sum()
    else:
        ground_truth = np.array(channel_importance, dtype=float)
        ground_truth = ground_truth / ground_truth.sum()

    # Channel appearance probabilities (more important = more likely)
    base_probs = 0.3 + 0.7 * ground_truth  # range [0.3, 1.0]

    journeys = []
    for _ in range(n_journeys):
        length = rng.randint(min_journey_length, max_journey_length + 1)

        # Sample channels — higher importance channels are more likely
        probs = base_probs + rng.normal(0, noise, n_channels)
        probs = np.clip(probs, 0.05, 1.0)

        # For each position, sample a channel
        journey = []
        for _pos in range(length):
            p = probs / probs.sum()
            ch = rng.choice(channel_names, p=p)
            journey.append(ch)

            # Add interaction effects: if a synergistic pair was just added,
            # boost the partner's probability
            if interaction_effects is not None and len(journey) >= 2:
                prev = journey[-2]
                synergy_boost = interaction_effects * ground_truth[prev]
                probs[prev] += synergy_boost
                probs = np.clip(probs, 0.05, 2.0)

        journeys.append(journey)

    return journeys, ground_truth, channel_names
