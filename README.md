# Shapley Attribution

A scikit-learn compatible Python library for multi-touch attribution modeling using Shapley values, based on [Zhao et al. (2018)](https://arxiv.org/abs/1804.05327).

## Installation

```bash
pip install -e .

# With benchmark dependencies
pip install -e ".[benchmarks]"

# With dev/test dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from shapley_attribution import (
    MonteCarloShapleyAttribution,
    LinearAttribution,
    make_attribution_problem,
    attribution_summary,
)

# Generate synthetic data with known ground truth
journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=8, n_journeys=5000, random_state=42
)

# Fit a model (scikit-learn style) — pass conversion labels
model = MonteCarloShapleyAttribution(n_iter=2000, random_state=42)
model.fit(journeys, y=conversions)

# Get aggregate attribution scores
scores = model.get_attribution()        # dict: channel -> score
array  = model.get_attribution_array()  # numpy array aligned with model.channels_

# Per-journey attribution matrix
matrix = model.transform(journeys)      # shape (n_journeys, n_channels)

# Compare multiple models
linear = LinearAttribution().fit(journeys, y=conversions)
results = attribution_summary(
    {"MC Shapley": model.get_attribution_array(),
     "Linear": linear.get_attribution_array()},
    ground_truth
)
```

## Models

### Shapley-based

| Model | Class | Complexity | Best for |
|---|---|---|---|
| **Simplified Shapley** | `SimplifiedShapleyAttribution` | O(n_journeys x n_channels) | Exact values, <= 20 channels |
| **Ordered Shapley** | `OrderedShapleyAttribution` | O(n_journeys x n_positions x 2^n_channels) | Position-aware, <= 15 channels |
| **Monte Carlo Shapley** | `MonteCarloShapleyAttribution` | O(n_iter x n_channels x n_background) | Scalable to 100+ channels |

The MC Shapley model trains a gradient-boosted conversion model internally, then computes Shapley values over that model's predictions using permutation sampling with background marginalization (the same approach as [KernelSHAP](https://github.com/shap/shap)).

### Heuristic Baselines

| Model | Class | Rule |
|---|---|---|
| **First Touch** | `FirstTouchAttribution` | 100% to first channel |
| **Last Touch** | `LastTouchAttribution` | 100% to last channel |
| **Linear** | `LinearAttribution` | Equal credit across all touchpoints |
| **Time Decay** | `TimeDecayAttribution` | Exponential decay favoring recent touchpoints |
| **Position Based** | `PositionBasedAttribution` | 40/20/40 split (first/middle/last) |

## scikit-learn Compatibility

All models inherit from `sklearn.base.BaseEstimator` and `TransformerMixin`:

```python
from sklearn.base import clone

model = MonteCarloShapleyAttribution(n_iter=2000)
model.get_params()          # {'n_iter': 2000, 'random_state': None, ...}
model.set_params(n_iter=500)
cloned = clone(model)       # Deep copy with same params
```

## Conversion Labels

All models accept an optional `y` parameter with binary conversion labels:

```python
model.fit(journeys, y=conversions)  # 1=converted, 0=not
model.fit(journeys)                 # Legacy mode: all journeys assumed converting
```

MC Shapley uses these labels to train a conversion model, giving it a significant accuracy advantage over heuristic baselines. The heuristic baselines use the labels to attribute credit only to converting journeys.

## Evaluation Metrics

```python
from shapley_attribution.metrics import (
    normalized_mean_absolute_error,  # NMAE in [0, 2], lower is better
    rank_correlation,                # Spearman rho in [-1, 1], higher is better
    top_k_overlap,                   # Fraction of true top-k recovered
    attribution_summary,             # Compare multiple models at once
)
```

## Synthetic Data

```python
from shapley_attribution.datasets import make_attribution_problem

journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=10,
    n_journeys=5000,
    max_journey_length=8,
    interaction_effects=0.5,   # Pairwise synergy between channels
    base_conversion_rate=0.3,
    random_state=42,
)
```

The synthetic generator creates journeys with roughly uniform channel sampling but conversion probability driven by channel presence and pairwise interactions via a logistic model with known coefficients. This rewards models that capture marginal contribution (Shapley) over frequency counting (heuristics).

## Benchmark

```bash
python benchmarks/benchmark.py
```

Sample output:

```
Model                         NMAE  Rank Corr    Top-3  Time(s)
---------------------------------------------------------------
First Touch                 0.0376     0.9048     0.67    0.003
Last Touch                  0.0373     0.9524     0.67    0.003
Linear                      0.0384     0.9048     0.67    0.004
Time Decay (0.5)            0.0384     0.9524     0.67    0.006
Position Based              0.0375     0.9524     0.67    0.026
Simplified Shapley          0.0387     0.9048     0.67    0.005
MC Shapley                  0.0278     0.9048     0.67    0.257

SCALABILITY TEST
    5 channels: MC=0.091s   Exact=0.002s
   10 channels: MC=0.304s   Exact=0.003s
   20 channels: MC=1.544s   Exact=skipped
   50 channels: MC=4.255s   Exact=skipped
```

MC Shapley achieves the lowest NMAE (closest to ground truth) by learning a conversion model and computing proper marginal contributions.

## Tests

```bash
pytest tests/ -v
```

110 tests covering sklearn API compliance, attribution correctness, MC convergence, input validation, and the synthetic dataset generator.

## Project Structure

```
shapley_attribution/
├── __init__.py                   # Public API
├── base.py                       # BaseAttributionModel (sklearn mixin)
├── models/
│   ├── simplified.py             # Exact set-based Shapley
│   ├── ordered.py                # Exact position-aware Shapley
│   └── monte_carlo.py            # Approximate Shapley (GBM + MC sampling)
├── baselines/
│   └── heuristic.py              # First/Last Touch, Linear, Time Decay, Position
├── datasets/
│   └── synthetic.py              # make_attribution_problem()
└── metrics/
    └── evaluation.py             # NMAE, rank correlation, top-k overlap
```

## References

- Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). [Shapley Value Methods for Attribution Modeling in Online Advertising](https://arxiv.org/abs/1804.05327). arXiv:1804.05327.
- Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research*.
- Lundberg, S., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://github.com/shap/shap). NeurIPS 2017.

## License

MIT
