from random import uniform
from dataclasses import dataclass
from typing import List, Tuple, Callable
from NeuralNetwork.dataStructure.ActivationFunction import ActivationFunction
from NeuralNetwork.dataStructure.ClassifiedData import ClassifiedData
from NeuralNetwork.dataStructure.PerformanceMeasurement import PerformanceMeasurement


@dataclass
class Neuron:
    weights: List[float]
    output: float
    product_value: float
    delta: float

    def product(self, inputs: List[float]) -> float:
        return self.weights[-1] + sum(
            self.weights[i] * inputs[i]
            for i in range(len(self.weights)-1)
        )

    @staticmethod
    def random(previous_layer_size: int) -> 'Neuron':
        weights = [
            uniform(-1, 1)
            for _ in range(previous_layer_size)
        ]
        return Neuron(weights=weights, output=0.0, delta=0.0, product_value=0.0)


@dataclass
class Layer:
    neurons: List[Neuron]

    @staticmethod
    def random(total_neurons: int, previous_layer_size: int) -> 'Layer':
        neurons = [
            Neuron.random(previous_layer_size)
            for _ in range(total_neurons)
        ]
        return Layer(neurons=neurons)


class NeuralNetwork:
    # layers_size includes input layer's size
    def __init__(self, layers_size: Tuple[int, ...], activation: ActivationFunction, error_function: Callable[[float, float], float]):
        self.layers_size = layers_size
        self.error_function = error_function
        self.activation = activation
        self.network: List[Layer] = []

        for i in range(1, len(layers_size)):
            current_layer_size = layers_size[i]
            if i != len(layers_size) - 1:
                current_layer_size += 1
            prev_layer_size = layers_size[i-1] + 1

            self.network.append(
                Layer.random(total_neurons=current_layer_size, previous_layer_size=prev_layer_size)
            )

    def apply_input(self, inputs: List[float]):
        for layer in self.network:
            inputs_storage = []
            for neuron in layer.neurons:
                product = neuron.product(inputs)
                output = self.activation.function(product)
                neuron.product_value = product
                neuron.output = output
                inputs_storage.append(output)
            inputs = inputs_storage

        return inputs

    def backpropagation(self, expected: List[float]):
        network = self.network

        output_neuron = network[-1].neurons[0]
        error = expected[0] - output_neuron.output
        derivative = self.activation.derivative(output_neuron.product_value)
        output_neuron.delta = error * derivative

        for layer_index in reversed(range(len(network)-1)):
            layer = network[layer_index]

            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                error = sum(
                    next_layer_neuron.weights[j] * next_layer_neuron.delta
                    for next_layer_neuron in network[layer_index + 1].neurons
                )
                derivative = self.activation.derivative(neuron.product_value)
                neuron.delta = error * derivative

    def update_weights(self, rate: float, inputs: List[float]):
        for layout_index, layout in enumerate(self.network):
            if layout_index != 0:
                # Setting the inputs as previous neuron's output
                inputs = [neuron.output for neuron in self.network[layout_index-1].neurons]

            for neuron in layout.neurons:
                for weight_index in range(len(inputs)):
                    neuron.weights[weight_index] += rate * neuron.delta * inputs[weight_index]
                neuron.weights[-1] += rate * neuron.delta

    def train(self, rate: float, epoch: int, training_data: List[ClassifiedData], testing_data: List[ClassifiedData]) -> PerformanceMeasurement:
        performance = PerformanceMeasurement()
        for epoch_i in range(epoch):
            total_training_errors = 0
            for data in training_data:
                outputs = self.apply_input(data.value)
                expected = [data.expected]
                total_training_errors += sum(
                    self.error_function(expected[i], outputs[i])
                    for i in range(len(expected))
                )
                self.backpropagation(expected)
                self.update_weights(rate, data.value)
            performance.add_training_error(total_training_errors)

            total_testing_errors = 0
            for data in testing_data:
                outputs = self.apply_input(data.value)
                expected = [data.expected]
                total_testing_errors += sum(
                    self.error_function(expected[i], outputs[i])
                    for i in range(len(expected))
                )
            performance.add_test_error(total_testing_errors)

            print("Epoch={} | Training Error={:.4f} | Testing Error={:.4f} ".format(epoch_i, total_training_errors, total_testing_errors))

        return performance

    def apply(self, inputs):
        outputs = self.apply_input(inputs)
        return outputs[0]
