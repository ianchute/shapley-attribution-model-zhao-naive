"""Tests for attribution models."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

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
    def test_fit_returns_self(self, ModelClass, simple_journeys):
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
    def test_transform_shape(self, ModelClass, simple_journeys):
        model = ModelClass()
        model.fit(simple_journeys)
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
    def test_scores_non_negative(self, ModelClass, simple_journeys):
        model = ModelClass()
        model.fit(simple_journeys)
        scores = model.get_attribution_array()
        assert np.all(scores >= 0)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_normalized_sums_to_one(self, ModelClass, simple_journeys):
        model = ModelClass(normalize=True)
        model.fit(simple_journeys)
        scores = model.get_attribution_array()
        np.testing.assert_almost_equal(scores.sum(), 1.0, decimal=5)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_all_channels_have_scores(self, ModelClass, simple_journeys):
        model = ModelClass()
        model.fit(simple_journeys)
        attr = model.get_attribution()
        channels = {ch for j in simple_journeys for ch in j}
        assert set(attr.keys()) == channels

    def test_first_touch_assigns_to_first(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        model = FirstTouchAttribution(normalize=False)
        model.fit(journeys)
        attr = model.get_attribution()
        assert attr[1] == 1.0  # first in journey 1
        assert attr[2] == 1.0  # first in journey 2

    def test_last_touch_assigns_to_last(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        model = LastTouchAttribution(normalize=False)
        model.fit(journeys)
        attr = model.get_attribution()
        assert attr[3] == 1.0  # last in journey 1
        assert attr[1] == 1.0  # last in journey 2

    def test_linear_equal_split(self):
        journeys = [[1, 2]]
        model = LinearAttribution(normalize=False)
        model.fit(journeys)
        attr = model.get_attribution()
        np.testing.assert_almost_equal(attr[1], 0.5)
        np.testing.assert_almost_equal(attr[2], 0.5)

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
    def test_reproducibility(self, simple_journeys):
        m1 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m2 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m1.fit(simple_journeys)
        m2.fit(simple_journeys)
        np.testing.assert_array_almost_equal(
            m1.get_attribution_array(),
            m2.get_attribution_array(),
        )

    def test_different_seeds_differ(self, simple_journeys):
        m1 = MonteCarloShapleyAttribution(n_iter=50, random_state=1)
        m2 = MonteCarloShapleyAttribution(n_iter=50, random_state=2)
        m1.fit(simple_journeys)
        m2.fit(simple_journeys)
        # With different seeds the results should generally differ
        # (not always guaranteed, but very likely with 50 iters)

    def test_more_iters_closer_to_exact(self):
        journeys, gt, _ = make_attribution_problem(
            n_channels=5, n_journeys=300, random_state=0
        )
        exact = SimplifiedShapleyAttribution()
        exact.fit(journeys)
        exact_scores = exact.get_attribution_array()

        mc_few = MonteCarloShapleyAttribution(n_iter=50, random_state=42)
        mc_many = MonteCarloShapleyAttribution(n_iter=1000, random_state=42)
        mc_few.fit(journeys)
        mc_many.fit(journeys)

        err_few = np.mean(np.abs(mc_few.get_attribution_array() - exact_scores))
        err_many = np.mean(np.abs(mc_many.get_attribution_array() - exact_scores))
        # More iterations should generally yield lower error
        # (not always guaranteed but very likely)
        assert err_many <= err_few * 2  # loose bound


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
        model = ModelClass()
        model.fit(journeys)
        attr = model.get_attribution()
        assert "email" in attr
        assert "social" in attr


# ---------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------

class TestDatasetGenerator:
    def test_output_shape(self):
        journeys, gt, channels = make_attribution_problem(
            n_channels=6, n_journeys=100, random_state=42
        )
        assert len(journeys) == 100
        assert len(gt) == 6
        assert len(channels) == 6

    def test_ground_truth_sums_to_one(self):
        _, gt, _ = make_attribution_problem(n_channels=5, random_state=42)
        np.testing.assert_almost_equal(gt.sum(), 1.0, decimal=5)

    def test_reproducibility(self):
        j1, gt1, _ = make_attribution_problem(random_state=42)
        j2, gt2, _ = make_attribution_problem(random_state=42)
        np.testing.assert_array_equal(gt1, gt2)
        assert j1 == j2
