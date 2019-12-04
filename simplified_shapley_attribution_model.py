from itertools import chain, combinations
from tqdm import tqdm
from collections import Counter


class SimplifiedShapleyAttributionModel:
    def powerset(self, x):
        s = list(x)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _phi(self, channel_index):
        S_channel = [k for k in self.journeys.keys() if channel_index in k]
        score = 0
        print(f"Computing phi for channel {channel_index}...")
        for S in tqdm(S_channel):
            score += self.journeys[S] / len(S)
        print(f"Attribution score for channel {channel_index}: {score:.2f}")
        print()
        return score

    def attribute(self, journeys):
        self.P = set(chain(*journeys))
        print("Running Simplified Shapley Attribution Model...")
        print(f"Found {len(self.P)} unique channels!")
        
        print("Computing journey statistics...")
        self.journeys = Counter([frozenset(journey) for journey in journeys])
        
        print(f"Computing attributions...")
        print()
        return {j: self._phi(j) for j in self.P}