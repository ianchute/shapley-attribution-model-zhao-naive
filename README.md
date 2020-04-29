## Shapley Value Methods for Attribution Modeling (Naive, Set-based)

A Python implementation of ["Shapley Value Methods for Attribution Modeling in Online Advertising" by Zhao, et al.](https://arxiv.org/abs/1804.05327)

**How to use:**

1. Clone this repository
```bash
git clone https://github.com/ianchute/shapley-attribution-model.git
```

2. Import one of the two Shapley Attribution Models
```python
  from simplified_shapley_attribution_model import SimplifiedShapleyAttributionModel
```

3. Initialize the model
```python
model = SimplifiedShapleyAttributionModel()
```

4. Feed customer journeys into the model (represented by list of lists of integers, where each integer represents a channel, product, or other object that can be attributed to a certain event). Sample data can be found in the `data` folder.
```python
import json
with open("data/sample.json", "r") as f:
  journeys = json.load(f)
result = model.attribute(journeys)
```

5. The result is a dictionary of attributions (keys are channels, values are attribution scores; for `OrderedShapleyAttributionModel`, a list of attributions is returned - one for each touchpoint in the journey)

**Available models:**

1. `SimplifiedShapleyAttributionModel`
2. `OrderedShapleyAttributionModel`
