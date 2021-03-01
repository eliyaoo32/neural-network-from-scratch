import cellpylib as cpl
import numpy as np


MAX_CYCLES = 80
BOARD_WIDTH = 16
BOARD_HEIGHT = 16


def init_configuration(width=BOARD_WIDTH, height=BOARD_HEIGHT):
    return cpl.init_simple2d(width, height, val=0)


def evolve_configuration(configuration, steps=1):
    return cpl.evolve2d(configuration, timesteps=steps, neighbourhood='Moore', apply_rule=cpl.game_of_life_rule)


def is_cyclic_configuration(configuration):
    configuration = evolve_configuration(configuration, steps=MAX_CYCLES)

    start_config = configuration[0]
    for i in range(1, MAX_CYCLES):
        current_config = configuration[i]
        if np.array_equal(start_config, current_config):
            return True

    return False

