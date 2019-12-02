from itertools import chain, combinations
from tqdm import tqdm


class SimplifiedShapleyAttributionModel:
    def __init__(self):
        self.P = set()

    def powerset(self, x):
        s = list(x)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _r(self, S):
        return sum(
            [1 if S == journey else 0 for journey in self.indexed_journeys[len(S)]]
        )

    def _phi(self, channel_index):
        S_all = [set(S) for S in self.P_power if channel_index in S]
        score = 0
        print(f"Computing phi for channel {channel_index}...")
        for S in tqdm(S_all):
            score += self._r(S) / len(S)
        print(f"Attribution score for channel {channel_index}: {score:.2f}")
        print()
        return score

    def attribute(self, journeys):
        self.P = set(chain(*journeys))
        print("Running Simplified Shapley Attribution Model...")
        print(f"Found {len(self.P)} unique channels!")
        self.P_power = list(self.powerset(self.P))
        self.journeys = [set(journey) for journey in journeys]
        self.indexed_journeys = {
            i: [S for S in self.journeys if len(S) == i]
            for i in range(1, len(self.P) + 1)
        }
        print(f"Proceeding to attribution computation...")
        print()
        return {j: self._phi(j) for j in self.P}
