from simplified_shapley_attribution_model import SimplifiedShapleyAttributionModel
import json

with open("data/sample.json", "r") as f:
    journeys = json.load(f)

o = SimplifiedShapleyAttributionModel()
result = o.attribute(journeys)

print(f"Total value: {len(journeys)}")
total = 0
for k, v in result.items():
    print(f"Channel {k}: {v:.2f}")
    total += v

print(f"Total of attributed values: {total:.2f}")
