from glob import glob
from typing import List
from .game import is_cyclic_configuration
from .dataSetTools import create_configuration_from_cells_format, read_file, configuration_to_neural_input, random_configuration
from NeuralNetwork.dataStructure.ClassifiedData import ClassifiedData


_encoded_configurations = [
    read_file(file)
    for file in glob("./data/*.cells")
]

_random_maps = []

# Add configurations from encoded files
_cyclic_configurations = [
    create_configuration_from_cells_format(encoded_con)
    for encoded_con in _encoded_configurations
]
# Add random configurations
_random_configuration = [
    random_configuration(0.1)
    for _ in range(20)
]

print("Starting Loading Data...")
training_data = _cyclic_configurations[:-10] + _random_configuration[:-10]
test_data = _cyclic_configurations[-10:] + _random_configuration[-10:]

classified_training_data: List[ClassifiedData] = [
    ClassifiedData(
        value=configuration_to_neural_input(config),
        expected=1.0 if is_cyclic_configuration(config) else -1.0
    )
    for config in training_data
]

classified_test_data: List[ClassifiedData] = [
    ClassifiedData(
        value=configuration_to_neural_input(config),
        expected=1.0 if is_cyclic_configuration(config) else -1.0
    )
    for config in test_data
]
print("Done Loading Data!")
