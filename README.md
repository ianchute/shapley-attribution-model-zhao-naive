# Shapley Value Methods for Attribution Modeling
## By: Ian Herve U. Chu Te

A Python implementation of ["Shapley Value Methods for Attribution Modeling in Online Advertising" by Zhao, et al.](https://arxiv.org/abs/1804.05327)

**How to use:**
1. `git clone https://github.com/ianchute/shapley-attribution-model.git`
2. Import one of the two Shapley Attribution Models
  e.g. `from simplified_shapley_attribution_model import SimplifiedShapleyAttributionModel`
3. Initialize the model
  e.g. `model = SimplifiedShapleyAttributionModel()`
4. Feed customer journeys into the model (represented by list of lists of integers, where each integer represents a channel, product, or other object that can be attributed to a certain event.
  e.g. `result = model.attribute(journeys)`
5. The result is a dictionary of attributions (keys are channels, values is the attribution; for OrderedShapleyAttributionModel, a list of attributions is returned - one for each touchpoint in the journey)
