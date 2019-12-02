from itertools import chain, combinations
from tqdm import tqdm


class OrderedShapleyAttributionModel:
    def __init__(self):
        self.P = set()

    def powerset(self, x):
        s = list(x)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _r(self, S, channel_index, touchpoint_index):
        return sum(
            [
                1 / journey.count(channel_index)
                if (
                    (S == journey_set)
                    and (journey[touchpoint_index - 1] == channel_index)
                )
                else 0
                for journey, journey_set in self.indexed_journeys[len(S)]
                if touchpoint_index <= len(journey)
            ]
        )

    def _phi(self, channel_index, touchpoint_index):
        S_all = [set(S) for S in self.P_power if channel_index in S]
        score = 0
        print(
            f"Computing phi for channel {channel_index}, touchpoint {touchpoint_index}..."
        )
        for S in tqdm(S_all):
            score += self._r(S, channel_index, touchpoint_index) / len(S)
        print(
            f"Attribution score for channel {channel_index}, touchpoint {touchpoint_index}: {score:.2f}"
        )
        print()
        return score

    def attribute(self, journeys):
        self.P = set(chain(*journeys))
        print("Running Ordered Shapley Attribution Model...")
        print(f"Found {len(self.P)} unique channels!")
        self.P_power = list(self.powerset(self.P))
        self.N = max([len(journey) for journey in journeys])
        print(f"Found {self.N} maximum touchpoints!")
        self.journeys = journeys
        self.indexed_journeys = {
            i: [(S, set(S)) for S in self.journeys if len(set(S)) == i]
            for i in range(1, len(self.P) + 1)
        }
        print(f"Proceeding to attribution computation...")
        print()
        return {j: [self._phi(j, i) for i in range(1, self.N + 1)] for j in self.P}
