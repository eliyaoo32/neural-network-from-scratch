import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class ActivationFunction:
    function: Callable[[float], float]
    derivative: Callable[[float], float]


def _sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


SIGMOID: ActivationFunction = ActivationFunction(
    function=_sigmoid_func,
    derivative=lambda x: _sigmoid_func(x) * (1 - _sigmoid_func(x))
)


ReLU: ActivationFunction = ActivationFunction(
    function=lambda x: max(x, 0),
    derivative=lambda x: 1 if x > 0 else 0
)

BINARY: ActivationFunction = ActivationFunction(
    function=lambda x: 1 if x >= 0 else 0,
    derivative=lambda x: 0,
)


def tanh_by_beta(beta: float) -> ActivationFunction:
    return ActivationFunction(
        function=lambda x: np.tanh(beta*x),
        derivative=lambda x: beta * (1-(np.tanh(beta*x)**2))
    )


TanH: ActivationFunction = tanh_by_beta(1.0)


def square_error(expected: float, outputs: float) -> float:
    return (expected-outputs)**2
