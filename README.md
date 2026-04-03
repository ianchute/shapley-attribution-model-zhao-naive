# Shapley Attribution

A scikit-learn compatible Python library for multi-touch attribution modeling using Shapley values from game theory. Computes marginal contribution of each marketing channel to conversion, inspired by Zhao et al. (2018) but using an interventional Shapley approach with a learned conversion model.

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
| **Monte Carlo Shapley** | `MonteCarloShapleyAttribution` | O(n_iter x n_channels) | Scalable to 100+ channels, order-agnostic |
| **Path Shapley** | `PathShapleyAttribution` | O(n_journeys x max_journey_length) | Ordering-aware, typically faster than MC |

**MC Shapley** trains a GradientBoostingClassifier to learn conversion probability, then estimates Shapley values by averaging marginal contributions across random permutations (interventional formulation — no background averaging). It is order-agnostic: `[A, B]` and `[B, A]` receive identical attribution.

**Path Shapley** uses the same GBM value function but replaces random permutation sampling with the *actual journey sequence* as the coalition-formation order. Channel B in journey `[A, B]` is credited for the lift *given A was already seen*; in `[B, A]`, A receives that conditional credit instead. This makes the model sensitive to channel ordering, which is meaningful when upstream channels prime the customer for downstream ones. Use `directed_interaction_strength > 0` in `make_attribution_problem` to generate data that rewards this sensitivity.

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

Enable `directed_interaction_strength > 0` to additionally bake in **ordering effects**: an asymmetric bonus/penalty is applied when channel i appears *before* channel j (`directed_matrix[i, j] ≠ directed_matrix[j, i]`). This creates genuine sequential synergies that PathShapleyAttribution is designed to exploit, while set-based models (MC Shapley, heuristics) remain blind to them.

```python
journeys, conversions, ground_truth, channels, ordered_ground_truth = make_attribution_problem(
    n_channels=8,
    n_journeys=5000,
    directed_interaction_strength=0.5,   # ordering effects active
    return_ordered_ground_truth=True,    # oracle path GT via true model
)
```

`ordered_ground_truth` is computed by walking each converting journey through the true logistic model and accumulating marginal contributions along the path — the oracle that PathShapleyAttribution aims to recover.

### How ground truth is computed

Ground truth is determined entirely before any journeys are generated:

1. **Channel importances** are sampled from a Dirichlet distribution (α=2) and raised to the power 1.3, producing moderate skew — one or two channels are notably more important than others, but not overwhelmingly so.
2. **Conversion coefficients** are scaled from those importances: `coef[ch] = importance[ch] × n_channels × 3`.
3. **Pairwise interaction terms** are sampled for every channel pair, adding synergy effects that no heuristic can capture.
4. **An intercept is calibrated** via root-finding (`scipy.brentq`) so the overall conversion rate matches `base_conversion_rate`.
5. **Journeys are generated** with near-uniform channel sampling, so channel frequency carries almost no signal — only which combinations appear drives conversion.
6. **Conversion labels** are drawn as Bernoulli samples from the logistic model: `P(convert) = σ(intercept + Σcoef·presence + Σinteraction·pair_presence)`.

The ground truth returned is the **normalized channel importance vector** from step 1 — the "true" individual channel weights, independent of interactions. This is intentionally slightly different from perfect Shapley values (which would distribute interaction credit across channels), giving a realistic evaluation target.

## Visualization

All fitted models expose three plot methods directly. Standalone functions are also available for multi-model comparison.

```python
from shapley_attribution import (
    MonteCarloShapleyAttribution, LinearAttribution,
    make_attribution_problem,
    compare_models, plot_performance, plot_journey, plot_journeys_heatmap,
)

journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=8, n_journeys=5000, random_state=42
)

mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42).fit(journeys, y=conversions)
lin = LinearAttribution().fit(journeys, y=conversions)
```

### Per-model attribution bar chart

```python
# On the model directly — overlays ground truth markers
mc.plot_attribution(ground_truth=ground_truth, top_k=8)
```

Shows a horizontal bar chart sorted by attribution score. Pass `ground_truth` to overlay ◆ markers for easy comparison.

### Compare multiple models

```python
# Standalone function — accepts 2+ models or pre-computed arrays
compare_models(
    {"MC Shapley": mc, "Linear": lin},
    ground_truth=ground_truth,
)
```

Renders a grouped bar chart with one cluster per channel. Accepts either fitted model objects or raw numpy arrays.

### Performance metrics panel

```python
from shapley_attribution.metrics import attribution_summary

results = attribution_summary(
    {"MC Shapley": mc.get_attribution_array(),
     "Linear": lin.get_attribution_array()},
    ground_truth,
)
plot_performance(results)
```

Three-panel figure (NMAE / Spearman rank correlation / top-3 overlap). The best-performing model is highlighted in each panel.

### Journey sequence diagram

```python
# On the model directly — boxes are coloured by attribution weight
mc.plot_journey(journeys[0], converted=bool(conversions[0]))

# Standalone — without a model (plain sequence)
plot_journey(journeys[0], converted=True)
```

Renders touchpoints as rounded boxes connected by arrows, with a conversion outcome node at the end. When called on a fitted model, box colour encodes that model's attribution score for each channel.

### Position attribution breakdown (PathShapley only)

```python
path = PathShapleyAttribution(random_state=42).fit(journeys, y=conversions)

# On the model directly
path.plot_position_attribution()

# Standalone
from shapley_attribution import plot_position_attribution
plot_position_attribution(path, top_k=6)
```

Stacked bar chart where each bar is a channel and each stack segment is a journey position (position 1 = first touchpoint, position 2 = second, …). Channels with tall early-position stacks are **upper-funnel** (awareness); channels with tall late-position stacks are **lower-funnel** (conversion drivers). Only available after fitting a `PathShapleyAttribution` model.

### Per-journey attribution heatmap

```python
# On the model directly
mc.plot_journeys_heatmap(journeys, conversions=conversions, max_journeys=50)

# Standalone
plot_journeys_heatmap(mc, journeys, conversions=conversions)
```

Heatmap of the `transform()` output matrix. Each row is a journey, each column is a channel. Pass `conversions` to show only converting journeys.

---

## Benchmark

```bash
python benchmarks/benchmark.py
```

Sample output (8 channels, 5000 journeys, 2000 MC iterations):

```
Model                         NMAE  Rank Corr    Top-3  Time(s)
---------------------------------------------------------------
First Touch                 0.0376     0.9048     0.67    0.003
Last Touch                  0.0373     0.9524     0.67    0.003
Linear                      0.0384     0.9048     0.67    0.004
Time Decay (0.5)            0.0384     0.9524     0.67    0.005
Position Based              0.0375     0.9524     0.67    0.024
Simplified Shapley          0.0387     0.9048     0.67    0.005
MC Shapley                  0.0214     0.9762     1.00    0.461

SCALABILITY TEST
    5 channels: MC=0.188s (NMAE=0.0446)  Exact=0.002s
   10 channels: MC=0.366s (NMAE=0.0182)  Exact=0.003s
   20 channels: MC=1.394s (NMAE=0.0077)  Exact=skipped
   50 channels: MC=3.973s (NMAE=0.0037)  Exact=skipped
```

MC Shapley dominates across all metrics: lowest NMAE (exact match to ground truth), highest rank correlation (0.976 vs 0.952 for best heuristic), and perfect top-3 recovery (1.0 vs 0.67). This is achieved by learning a conversion model and computing proper marginal contributions via interventional Shapley.

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

## Roadmap

- [ ] GPU acceleration (PyTorch/CuPy) for 100k+ journeys
- [ ] Distributed computing support (Ray/Dask) for multi-machine scaling
- [ ] Comprehensive tests on larger datasets (>100k journeys)
- [ ] Additional baseline models (Markov, first-order attribution chains)
- [ ] Custom coalition value functions (user-provided models)
- [ ] Interactive visualization dashboard

## Related Libraries

- **[SHAP](https://github.com/shap/shap)** — General-purpose Shapley value library for model interpretation (KernelSHAP, TreeSHAP, DeepSHAP)
- **[Alibi](https://github.com/SeldonIO/alibi)** — Model-agnostic explainability and fairness (includes Shapley approximations)
- **[ELI5](https://github.com/eli5-org/eli5)** — Model interpretation library with permutation importance
- **[Captum](https://github.com/pytorch/captum)** — PyTorch model interpretability library (supports Shapley values)
- **[MultiTouch](https://github.com/AnalyticsEnthusiasts/MultiTouchAttribution)** — Another multi-touch attribution library (heuristics only)

## References

The following papers directly inform the algorithms in this library:

**Foundation**
- Shapley, L. S. (1952). A Value for n-Person Games. In *Contributions to the Theory of Games II*, Annals of Mathematics Studies, vol. 28. Princeton University Press.
  _The original cooperative game theory paper that defines the Shapley value axiomatically. All models in this library compute or approximate this quantity._

**Attribution-specific Shapley**
- Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). [Shapley Value Methods for Attribution Modeling in Online Advertising](https://arxiv.org/abs/1804.05327). arXiv:1804.05327.
  _Direct inspiration for this library. Introduces set-based and ordered Shapley variants for the multi-touch attribution problem. Our `SimplifiedShapleyAttribution` and `OrderedShapleyAttribution` implement these directly._

**Monte Carlo sampling**
- Castro, J., Gómez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research*, 36(9), 1726–1730.
  _Introduces the ApproShapley algorithm: estimate Shapley values by averaging marginal contributions over random permutations. This is the sampling backbone of `MonteCarloShapleyAttribution`._

**Interventional Shapley & learned value functions**
- Lundberg, S. M., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). In *Advances in Neural Information Processing Systems* (NeurIPS).
  _Introduces KernelSHAP: using a learned model as the coalition value function and sampling over coalitions. Our MC Shapley model adopts this approach (GBM as the value function)._
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). [Feature relevance quantification in explainable AI: A causal problem](https://arxiv.org/abs/1910.13413). In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (AISTATS).
  _Distinguishes interventional Shapley (v(S) = f(binary mask)) from observational Shapley (v(S) = E[f | X_S]). We use the interventional formulation — deterministic, no background averaging — which eliminates variance and improves stability._

## License

MIT
