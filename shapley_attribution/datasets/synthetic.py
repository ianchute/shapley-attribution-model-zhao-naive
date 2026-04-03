"""
Synthetic data generators for attribution model benchmarking.

Generates customer journeys with a known ground-truth conversion model
so that attribution methods can be evaluated against true channel importance.

Key design: all channels appear with roughly equal frequency, but
conversion depends on which channels appear *together* — and, when
directed_interaction_strength > 0, in what *order*.  This lets us
benchmark both set-based models (MC Shapley) and ordering-aware models
(PathShapleyAttribution) against appropriate ground truths.
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
    directed_interaction_strength=0.0,
    noise=0.1,
    base_conversion_rate=0.3,
    random_state=None,
    return_ordered_ground_truth=False,
):
    """Generate a synthetic attribution problem with known ground truth.

    Creates customer journeys where conversion probability depends on
    channel presence through a logistic model with known coefficients,
    pairwise interaction terms, and optionally directed (asymmetric)
    sequential synergies.

    Crucially, channel *sampling* probabilities are roughly uniform —
    the signal is in which combinations (and optionally, which orderings)
    drive conversion, not in which channels appear more often.

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
        importances are generated from a Dirichlet distribution.
    interaction_effects : float, default=0.5
        Strength of undirected (symmetric) pairwise synergy.  Higher
        values make channel combinations more important than individual
        presence alone.
    directed_interaction_strength : float, default=0.0
        Strength of directed (asymmetric) sequential synergies.
        When > 0, the conversion model rewards specific orderings:
        e.g., seeing channel A *then* channel B may have a different
        effect than seeing B then A.  Activates ordering effects that
        PathShapleyAttribution is designed to capture.
    noise : float, default=0.1
        Noise in channel sampling probabilities.
    base_conversion_rate : float, default=0.3
        Target overall conversion rate.
    random_state : int or None, default=None
        Seed for reproducibility.
    return_ordered_ground_truth : bool, default=False
        If True, returns a 5-tuple including the oracle path ground truth —
        the true attribution under the directed model, computed by running
        the actual journey sequences through the ground-truth logistic model.
        Only meaningful when directed_interaction_strength > 0.

    Returns
    -------
    journeys : list of list of int
    conversions : ndarray of shape (n_journeys,)
    ground_truth : ndarray of shape (n_channels,)
        Normalized individual channel importance weights.  This is the
        appropriate ground truth for set-based models (MC Shapley,
        heuristics).
    channel_names : list of int
    ordered_ground_truth : ndarray of shape (n_channels,)  [only when return_ordered_ground_truth=True]
        Oracle path Shapley values computed using the true logistic model
        along actual journey sequences.  This is the appropriate ground
        truth for PathShapleyAttribution when directed interactions are on.
        Identical to ground_truth when directed_interaction_strength=0.
    """
    rng = np.random.RandomState(random_state)
    channel_names = list(range(n_channels))

    # ---- Channel importance (individual conversion coefficients) ----
    if channel_importance is None:
        raw = rng.dirichlet(np.ones(n_channels) * 2.0)
        raw = raw ** 1.3
        ground_truth = raw / raw.sum()
    else:
        ground_truth = np.array(channel_importance, dtype=float)
        ground_truth = ground_truth / ground_truth.sum()

    channel_coefs = ground_truth * n_channels * 3.0

    # ---- Undirected pairwise interaction matrix (symmetric) ----
    interaction_matrix = np.zeros((n_channels, n_channels))
    if interaction_effects > 0:
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                synergy = rng.uniform(-0.3, 1.0) * interaction_effects
                synergy *= (ground_truth[i] + ground_truth[j]) * n_channels
                interaction_matrix[i, j] = synergy
                interaction_matrix[j, i] = synergy

    # ---- Directed interaction matrix (asymmetric) ----
    # directed_matrix[i, j] = bonus when channel i appears BEFORE channel j.
    # A positive value means the sequence i → j is synergistic; negative
    # means that ordering is detrimental.  directed_matrix[i, j] ≠ directed_matrix[j, i]
    # by construction, creating genuine ordering effects.
    directed_matrix = np.zeros((n_channels, n_channels))
    if directed_interaction_strength > 0:
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                # Asymmetric: sample independently for each direction
                synergy = rng.uniform(-0.5, 1.0) * directed_interaction_strength
                synergy *= (ground_truth[i] + ground_truth[j]) * n_channels
                directed_matrix[i, j] = synergy

    # ---- Calibrate intercept for target conversion rate ----
    cal_logits = []
    for _ in range(2000):
        length = rng.randint(min_journey_length, max_journey_length + 1)
        journey = list(rng.choice(channel_names, size=length))
        logit_p = _compute_logit(
            journey, channel_coefs, interaction_matrix, directed_matrix, n_channels
        )
        cal_logits.append(logit_p)

    cal_logits = np.array(cal_logits)

    from scipy.optimize import brentq

    def _conv_rate(intercept):
        return expit(intercept + cal_logits).mean() - base_conversion_rate

    try:
        intercept = brentq(_conv_rate, -20, 20)
    except ValueError:
        intercept = -np.median(cal_logits)

    # ---- Generate journeys ----
    journeys = []
    conversions = []

    for _ in range(n_journeys):
        length = rng.randint(min_journey_length, max_journey_length + 1)
        probs = np.ones(n_channels) + rng.normal(0, noise, n_channels)
        probs = np.clip(probs, 0.3, 1.7)
        probs = probs / probs.sum()
        journey = list(rng.choice(channel_names, size=length, p=probs))
        journeys.append(journey)

        logit_p = intercept + _compute_logit(
            journey, channel_coefs, interaction_matrix, directed_matrix, n_channels
        )
        conv_prob = expit(logit_p)
        conversions.append(int(rng.random() < conv_prob))

    conversions = np.array(conversions)

    if not return_ordered_ground_truth:
        return journeys, conversions, ground_truth, channel_names

    # ---- Oracle path ground truth ----
    # For each converting journey, walk the actual sequence and compute
    # marginal contributions using the TRUE logistic model parameters.
    # This is the ground truth that PathShapleyAttribution aims to recover.
    ordered_gt = _compute_oracle_path_gt(
        journeys, conversions, channel_coefs, directed_matrix, intercept, n_channels
    )

    return journeys, conversions, ground_truth, channel_names, ordered_gt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_logit(journey, channel_coefs, interaction_matrix, directed_matrix, n_channels):
    """Compute the logit contribution (excluding intercept) for a journey."""
    present = np.zeros(n_channels)
    for ch in set(journey):
        present[ch] = 1.0

    logit_p = np.dot(channel_coefs, present)

    # Undirected pairwise interactions (symmetric, set-based)
    for i in range(n_channels):
        if present[i] == 0:
            continue
        for j in range(i + 1, n_channels):
            if present[j] == 0:
                continue
            logit_p += interaction_matrix[i, j]

    # Directed interactions (asymmetric, order-based)
    if directed_matrix.any():
        first_pos = {}
        for pos, ch in enumerate(journey):
            if ch not in first_pos:
                first_pos[ch] = pos
        for i in range(n_channels):
            if i not in first_pos:
                continue
            for j in range(n_channels):
                if j not in first_pos or i == j:
                    continue
                if first_pos[i] < first_pos[j]:  # i appears before j
                    logit_p += directed_matrix[i, j]

    return logit_p


def _compute_logit_ordered_coalition(ordered_coalition, channel_coefs,
                                     directed_matrix, intercept, n_channels):
    """Compute full logit for an ordered coalition (list of channels in sequence order)."""
    present = np.zeros(n_channels)
    for ch in ordered_coalition:
        present[ch] = 1.0

    logit_p = intercept + np.dot(channel_coefs, present)

    # Directed interactions: for each pair (a, b) where a precedes b in the coalition
    for p in range(len(ordered_coalition)):
        for q in range(p + 1, len(ordered_coalition)):
            a = ordered_coalition[p]  # a precedes b
            b = ordered_coalition[q]
            logit_p += directed_matrix[a, b]

    return logit_p


def _compute_oracle_path_gt(journeys, conversions, channel_coefs, directed_matrix,
                             intercept, n_channels):
    """Compute the oracle path Shapley ground truth using true model parameters.

    For each converting journey, walks the deduplicated channel sequence and
    computes marginal contributions against the true logistic model.  This is
    the ground truth that PathShapleyAttribution (using its GBM approximation)
    aims to recover.
    """
    path_attr = np.zeros(n_channels)

    for journey, converted in zip(journeys, conversions):
        if not converted:
            continue

        # Deduplicate preserving first-occurrence order
        seen = set()
        unique_seq = []
        for ch in journey:
            if ch not in seen:
                seen.add(ch)
                unique_seq.append(ch)

        # v({}) = sigmoid(intercept) — baseline with no channels
        prev_value = expit(intercept)
        coalition = []

        for ch in unique_seq:
            coalition.append(ch)
            current_logit = _compute_logit_ordered_coalition(
                coalition, channel_coefs, directed_matrix, intercept, n_channels
            )
            current_value = expit(current_logit)
            contribution = current_value - prev_value
            path_attr[ch] += max(contribution, 0.0)
            prev_value = current_value

    if path_attr.sum() > 0:
        path_attr = path_attr / path_attr.sum()

    return path_attr
