from dataclasses import dataclass, field
from typing import List


@dataclass
class PerformanceMeasurement:
    training_errors: List[float] = field(default_factory=list)
    test_errors: List[float] = field(default_factory=list)

    def add_training_error(self, error):
        self.training_errors.append(error)

    def add_test_error(self, error):
        self.test_errors.append(error)
