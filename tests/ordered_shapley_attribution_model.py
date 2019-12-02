from ordered_shapley_attribution_model import OrderedShapleyAttributionModel
import pandas as pd
import numpy as np

conversion_token = 21
journeys = pd.read_csv("data/sample.csv").values
journeys = [
    [
        int(channel)
        for channel in journey
        if not np.isnan(channel) and channel != conversion_token
    ]
    for journey in journeys
]

o = OrderedShapleyAttributionModel()
result = o.attribute(journeys)

print(f"Total value: {len(journeys)}")
total = 0
for k, v in result.items():
    print(f"Channel {k}: {v}")
#     total += v

# print(f"Total of attributed values: {total:.2f}")
