from ordered_shapley_attribution_model import OrderedShapleyAttributionModel
import json

with open("data/sample.json", "r") as f:
    journeys = json.load(f)

o = OrderedShapleyAttributionModel()
result = o.attribute(journeys)

print(f"Total value: {len(journeys)}")
total = 0
for k, v in result.items():
    vsum = sum(v)
    print(f"Channel {k}: {vsum}")
    total += vsum

print(f"Total of attributed values: {total:.2f}")
