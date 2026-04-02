"""Tests for attribution models."""

import numpy as np
import pytest

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


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def simple_journeys():
    return [[1, 2, 3], [1, 2], [2, 3], [1], [3, 1, 2]]


@pytest.fixture
def simple_conversions():
    return np.array([1, 1, 0, 1, 0])


@pytest.fixture
def synthetic_data():
    return make_attribution_problem(
        n_channels=5, n_journeys=500, random_state=42
    )


ALL_MODELS = [
    SimplifiedShapleyAttribution,
    MonteCarloShapleyAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
]


# ---------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------

class TestSklearnAPI:
    """Verify scikit-learn estimator contract."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_returns_self(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        result = model.fit(simple_journeys, y=simple_conversions)
        assert result is model

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_without_y(self, ModelClass, simple_journeys):
        """All models should work without y (legacy mode)."""
        model = ModelClass()
        result = model.fit(simple_journeys)
        assert result is model

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_get_params(self, ModelClass):
        model = ModelClass()
        params = model.get_params()
        assert isinstance(params, dict)
        assert "normalize" in params

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_set_params(self, ModelClass):
        model = ModelClass()
        model.set_params(normalize=False)
        assert model.normalize is False

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_clone(self, ModelClass):
        from sklearn.base import clone
        model = ModelClass()
        cloned = clone(model)
        assert type(cloned) is type(model)
        assert cloned.get_params() == model.get_params()

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_not_fitted_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_attribution()

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_transform_shape(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        result = model.transform(simple_journeys)
        assert result.shape == (len(simple_journeys), len(model.channels_))

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_transform(self, ModelClass, simple_journeys):
        model = ModelClass()
        result = model.fit_transform(simple_journeys)
        assert result.shape[0] == len(simple_journeys)


# ---------------------------------------------------------------
# Attribution correctness
# ---------------------------------------------------------------

class TestAttributionCorrectness:
    """Test that attribution scores are sensible."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_scores_non_negative(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        scores = model.get_attribution_array()
        assert np.all(scores >= 0)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_normalized_sums_to_one(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass(normalize=True)
        model.fit(simple_journeys, y=simple_conversions)
        scores = model.get_attribution_array()
        np.testing.assert_almost_equal(scores.sum(), 1.0, decimal=5)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_all_channels_discovered(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        channels = {ch for j in simple_journeys for ch in j}
        assert set(model.channels_) == channels

    def test_first_touch_assigns_to_first(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        y = [1, 1]
        model = FirstTouchAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert attr[1] == 1.0  # first in journey 1
        assert attr[2] == 1.0  # first in journey 2

    def test_last_touch_assigns_to_last(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        y = [1, 1]
        model = LastTouchAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert attr[3] == 1.0  # last in journey 1
        assert attr[1] == 1.0  # last in journey 2

    def test_linear_equal_split(self):
        journeys = [[1, 2]]
        y = [1]
        model = LinearAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        np.testing.assert_almost_equal(attr[1], 0.5)
        np.testing.assert_almost_equal(attr[2], 0.5)

    def test_non_converting_gets_no_credit(self):
        """Non-converting journeys should not receive attribution credit."""
        journeys = [[1, 2], [3, 4]]
        y = [1, 0]  # Only first journey converts
        model = LinearAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        # Channels 3 and 4 only appear in non-converting journey
        assert attr.get(3, 0.0) == 0.0
        assert attr.get(4, 0.0) == 0.0

    def test_simplified_shapley_single_channel(self):
        journeys = [[1], [1], [1]]
        model = SimplifiedShapleyAttribution()
        model.fit(journeys)
        attr = model.get_attribution()
        assert 1 in attr
        assert attr[1] > 0


# ---------------------------------------------------------------
# Monte Carlo convergence
# ---------------------------------------------------------------

class TestMonteCarlo:
    def test_reproducibility(self, simple_journeys, simple_conversions):
        m1 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m2 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m1.fit(simple_journeys, y=simple_conversions)
        m2.fit(simple_journeys, y=simple_conversions)
        np.testing.assert_array_almost_equal(
            m1.get_attribution_array(),
            m2.get_attribution_array(),
        )

    def test_uses_conversion_labels(self):
        """MC Shapley should give different results with different labels."""
        journeys = [[1, 2], [2, 3], [1, 3], [1, 2, 3]] * 50
        y1 = ([1, 0, 0, 1]) * 50  # ch 1+2 together convert
        y2 = ([0, 1, 0, 1]) * 50  # ch 2+3 together convert

        m1 = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
        m2 = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
        m1.fit(journeys, y=y1)
        m2.fit(journeys, y=y2)

        s1 = m1.get_attribution_array()
        s2 = m2.get_attribution_array()
        # The two should differ meaningfully
        assert not np.allclose(s1, s2, atol=0.05)

    def test_mc_beats_heuristics_on_synthetic(self):
        """MC Shapley should recover true importance better than heuristics."""
        journeys, conv, gt, _ = make_attribution_problem(
            n_channels=8, n_journeys=5000, random_state=42
        )

        mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42)
        mc.fit(journeys, y=conv)

        linear = LinearAttribution()
        linear.fit(journeys, y=conv)

        first = FirstTouchAttribution()
        first.fit(journeys, y=conv)

        from shapley_attribution.metrics import normalized_mean_absolute_error

        mc_nmae = normalized_mean_absolute_error(gt, mc.get_attribution_array())
        lin_nmae = normalized_mean_absolute_error(gt, linear.get_attribution_array())
        first_nmae = normalized_mean_absolute_error(gt, first.get_attribution_array())

        # MC Shapley should have lower NMAE (closer to ground truth)
        # than both heuristics
        assert mc_nmae < lin_nmae, (
            f"MC NMAE {mc_nmae:.4f} should be < Linear NMAE {lin_nmae:.4f}"
        )
        assert mc_nmae < first_nmae, (
            f"MC NMAE {mc_nmae:.4f} should be < First Touch NMAE {first_nmae:.4f}"
        )


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_empty_journeys_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(ValueError):
            model.fit([])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_empty_single_journey_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(ValueError):
            model.fit([[]])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_string_channels(self, ModelClass):
        journeys = [["email", "social"], ["social", "search"], ["email"]]
        y = [1, 0, 1]
        model = ModelClass()
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert "email" in attr or "social" in attr


# ---------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------

class TestDatasetGenerator:
    def test_output_shape(self):
        journeys, conv, gt, channels = make_attribution_problem(
            n_channels=6, n_journeys=100, random_state=42
        )
        assert len(journeys) == 100
        assert len(conv) == 100
        assert len(gt) == 6
        assert len(channels) == 6

    def test_has_both_classes(self):
        _, conv, _, _ = make_attribution_problem(
            n_channels=5, n_journeys=1000, random_state=42
        )
        assert conv.sum() > 0
        assert (1 - conv).sum() > 0

    def test_ground_truth_sums_to_one(self):
        _, _, gt, _ = make_attribution_problem(n_channels=5, random_state=42)
        np.testing.assert_almost_equal(gt.sum(), 1.0, decimal=5)

    def test_reproducibility(self):
        j1, c1, gt1, _ = make_attribution_problem(random_state=42)
        j2, c2, gt2, _ = make_attribution_problem(random_state=42)
        np.testing.assert_array_equal(gt1, gt2)
        np.testing.assert_array_equal(c1, c2)
        assert j1 == j2
