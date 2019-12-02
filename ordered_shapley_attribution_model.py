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
                1
                if (S == journey_set)
                and (journey[touchpoint_index - 1] == channel_index)
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
            score += 1.0 / len(S) * self._r(S, channel_index, touchpoint_index)
        return score

    def attribute(self, journeys):
        self.P = set(chain(*journeys))
        self.P_power = self.powerset(self.P)
        self.N = max([len(journey) for journey in journeys])
        self.journeys = journeys
        self.indexed_journeys = {
            i: [(S, set(S)) for S in self.journeys if len(set(S)) == i]
            for i in range(1, len(self.P) + 1)
        }

        return {j: [self._phi(j, i) for i in range(1, self.N + 1)] for j in self.P}
