#!/usr/bin/env python
"""
Benchmark: Compare all attribution models against synthetic ground truth.

Usage
-----
    python -m benchmarks.benchmark
    python benchmarks/benchmark.py

Generates synthetic journeys with converting and non-converting outcomes,
then measures how well each model recovers the true channel importances.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapley_attribution import (
    SimplifiedShapleyAttribution,
    MonteCarloShapleyAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
)
from shapley_attribution.datasets import make_attribution_problem
from shapley_attribution.metrics import attribution_summary, normalized_mean_absolute_error


def run_benchmark(
    n_channels=8,
    n_journeys=5000,
    max_journey_length=6,
    random_state=42,
    mc_iters=2000,
):
    """Run the full benchmark suite."""

    print("=" * 70)
    print("SHAPLEY ATTRIBUTION BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Channels:           {n_channels}")
    print(f"  Journeys:           {n_journeys}")
    print(f"  Max journey length: {max_journey_length}")
    print(f"  MC iterations:      {mc_iters}")
    print(f"  Random state:       {random_state}")

    # Generate synthetic data with known ground truth
    print("\nGenerating synthetic data...")
    journeys, conversions, ground_truth, channel_names = make_attribution_problem(
        n_channels=n_channels,
        n_journeys=n_journeys,
        max_journey_length=max_journey_length,
        interaction_effects=0.5,
        random_state=random_state,
    )

    n_converted = conversions.sum()
    print(f"  Ground truth:       {np.array2string(ground_truth, precision=4)}")
    print(f"  Top channel:        {channel_names[np.argmax(ground_truth)]}")
    print(f"  Conversion rate:    {conversions.mean():.1%} ({n_converted}/{n_journeys})")

    # Define models
    models = {
        "First Touch": FirstTouchAttribution(),
        "Last Touch": LastTouchAttribution(),
        "Linear": LinearAttribution(),
        "Time Decay (0.5)": TimeDecayAttribution(decay_rate=0.5),
        "Position Based": PositionBasedAttribution(),
        "Simplified Shapley": SimplifiedShapleyAttribution(),
        "MC Shapley": MonteCarloShapleyAttribution(
            n_iter=mc_iters, random_state=random_state
        ),
    }

    # Fit all models — pass y=conversions so models can use it
    attributions = {}
    timings = {}

    for name, model in models.items():
        print(f"\n  Fitting {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        model.fit(journeys, y=conversions)
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        print(f"({elapsed:.3f}s)")

        scores = model.get_attribution_array()
        attributions[name] = scores

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    summary = attribution_summary(attributions, ground_truth, channel_names)

    # Print table
    header = f"{'Model':<25} {'NMAE':>8} {'Rank Corr':>10} {'Top-3':>8} {'Time(s)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for name in models:
        m = summary[name]
        t = timings[name]
        print(
            f"{name:<25} {m['nmae']:>8.4f} {m['rank_correlation']:>10.4f} "
            f"{m['top_3_overlap']:>8.2f} {t:>8.3f}"
        )

    # Print channel-level attributions
    print(f"\n{'':=<70}")
    print("CHANNEL-LEVEL ATTRIBUTIONS (normalized)")
    print(f"{'':=<70}")

    gt_norm = ground_truth / ground_truth.sum()
    ch_header = f"{'Model':<25}" + "".join(f"{'Ch ' + str(c):>8}" for c in channel_names)
    print(f"\n{ch_header}")
    print("-" * len(ch_header))

    print(f"{'Ground Truth':<25}" + "".join(f"{v:>8.4f}" for v in gt_norm))
    for name in models:
        scores = attributions[name]
        if scores.sum() > 0:
            scores = scores / scores.sum()
        print(f"{name:<25}" + "".join(f"{v:>8.4f}" for v in scores))

    print(f"\n{'':=<70}")
    print("SCALABILITY TEST")
    print(f"{'':=<70}")

    # Test scalability with increasing channels
    for nc in [5, 10, 20, 50]:
        j, conv, gt, _ = make_attribution_problem(
            n_channels=nc, n_journeys=2000, random_state=42
        )

        mc = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
        t0 = time.perf_counter()
        mc.fit(j, y=conv)
        mc_time = time.perf_counter() - t0

        mc_scores = mc.get_attribution_array()
        mc_nmae = normalized_mean_absolute_error(gt, mc_scores)

        if nc <= 15:
            exact = SimplifiedShapleyAttribution()
            t0 = time.perf_counter()
            exact.fit(j, y=conv)
            exact_time = time.perf_counter() - t0
            exact_scores = exact.get_attribution_array()
            exact_nmae = normalized_mean_absolute_error(gt, exact_scores)
            print(
                f"  {nc:>3} channels: "
                f"MC={mc_time:.3f}s (NMAE={mc_nmae:.4f})  "
                f"Exact={exact_time:.3f}s (NMAE={exact_nmae:.4f})"
            )
        else:
            print(
                f"  {nc:>3} channels: "
                f"MC={mc_time:.3f}s (NMAE={mc_nmae:.4f})  "
                f"Exact=skipped (too many channels)"
            )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_benchmark()
