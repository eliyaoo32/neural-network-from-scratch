from dataclasses import dataclass
from typing import List


@dataclass
class ClassifiedData:
    value: List[int]
    expected: float

    def __repr__(self):
        return "<Value={}, Output=({})>".format(self.value, self.expected)
