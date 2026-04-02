"""
Evaluation metrics for comparing attribution models against ground truth.
"""

import numpy as np
from scipy import stats


def normalized_mean_absolute_error(y_true, y_pred):
    """Compute the normalized mean absolute error (NMAE).

    Both inputs are normalized to sum to 1 before comparison.

    Parameters
    ----------
    y_true : array-like of shape (n_channels,)
        Ground truth attribution weights.
    y_pred : array-like of shape (n_channels,)
        Predicted attribution weights.

    Returns
    -------
    nmae : float
        Value in [0, 2]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.sum() > 0:
        y_true = y_true / y_true.sum()
    if y_pred.sum() > 0:
        y_pred = y_pred / y_pred.sum()

    return np.mean(np.abs(y_true - y_pred))


def rank_correlation(y_true, y_pred):
    """Spearman rank correlation between true and predicted attributions.

    Measures whether the model ranks channels in the same order as the
    ground truth, regardless of the exact score magnitude.

    Parameters
    ----------
    y_true : array-like of shape (n_channels,)
    y_pred : array-like of shape (n_channels,)

    Returns
    -------
    rho : float
        Spearman correlation in [-1, 1].  Higher is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 3:
        # Spearman is not meaningful with < 3 items
        return float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else 1.0

    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def top_k_overlap(y_true, y_pred, k=3):
    """Fraction of the true top-k channels that appear in predicted top-k.

    Parameters
    ----------
    y_true : array-like of shape (n_channels,)
    y_pred : array-like of shape (n_channels,)
    k : int, default=3
        Number of top channels to compare.

    Returns
    -------
    overlap : float
        Value in [0, 1].  1.0 means perfect top-k agreement.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    k = min(k, len(y_true))

    true_topk = set(np.argsort(y_true)[-k:])
    pred_topk = set(np.argsort(y_pred)[-k:])

    return len(true_topk & pred_topk) / k


def attribution_summary(models_dict, ground_truth, channel_names=None):
    """Compare multiple models against ground truth.

    Parameters
    ----------
    models_dict : dict
        ``{model_name: attribution_array}`` where each array has shape
        ``(n_channels,)``.
    ground_truth : array-like of shape (n_channels,)
        True attribution weights.
    channel_names : list or None
        Channel identifiers for display.

    Returns
    -------
    summary : dict
        ``{model_name: {metric_name: value}}``.
    """
    gt = np.asarray(ground_truth, dtype=float)
    if gt.sum() > 0:
        gt_norm = gt / gt.sum()
    else:
        gt_norm = gt

    results = {}
    for name, pred in models_dict.items():
        pred = np.asarray(pred, dtype=float)
        if pred.sum() > 0:
            pred_norm = pred / pred.sum()
        else:
            pred_norm = pred

        results[name] = {
            "nmae": normalized_mean_absolute_error(gt_norm, pred_norm),
            "rank_correlation": rank_correlation(gt_norm, pred_norm),
            "top_3_overlap": top_k_overlap(gt_norm, pred_norm, k=3),
        }

    return results
