from random import random
from pathlib import Path
import numpy as np
from .game import init_configuration, BOARD_HEIGHT, BOARD_WIDTH


def create_configuration_from_cells_format(cells_format: str):
    configuration = init_configuration()
    lines = [
        line.strip()
        for line in cells_format.splitlines()
        if not line.startswith('!')
    ]

    for line_index, line in enumerate(lines):
        for word_index, word in enumerate(line):
            configuration[:, [line_index], [word_index]] = 1 if word == 'O' else 0

    return configuration


def read_file(file_path: str) -> str:
    return Path(file_path).read_text()


def configuration_to_neural_input(configuration):
    array1 = configuration[0].flatten()
    new_array = np.where(array1 == 0, -1, array1)
    return new_array


def random_configuration(live_probability: float):
    config = init_configuration()
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_HEIGHT):
            config[:, [i], [j]] = 1 if random() < live_probability else 0

    return config
