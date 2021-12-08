import math

WORLD_WIDTH, WORLD_HEIGHT = 100, 100
STEP_PER_GENERATION = 128
NUM_GENERATIONS = 1000
POPULATION = 1000

GENOME_LENGTH = 24  # Number of gene in a genome
GENE_LENGTH = 6  # Number of hex characters in a gene
NUM_ID_BITS = 5  # Number of bits dedicated to encoding source/sink id
MAX_INTERNAL_NEURON = 3
MAX_WEIGHT = (
    4  # Weights decoded from the genome will be scaled to (-MAX_WEIGHT, MAX_WEIGHT)
)
WEIGHT_SCALE = math.ceil(
    (2 ** (GENE_LENGTH * 4 - 2 * NUM_ID_BITS - 3)) / MAX_WEIGHT
)  # factor to scale weights

POINT_MUTATION_RATE = 0.001  # Probability for a bit flips in genomes (0, 1)
POPULATION_SENSOR_RADIUS = (
    2  # How far away creatures can sense other creature surronding
)
LONG_PROBE_DISTANCE = (
    24  # How far away creatures can look for barrier/creature in 1 directure
)
SHORT_PROBE_DISTANCE = 4
OSC_INITIAL_PERIOD = 32

NEURON_COLORS = {
    "sensory": "#42caff",
    "internal": "#8a8a8a",
    "action": "#ffb24d",
}  # colors for nodes in brain graphs

# Availabe neuron types
NEURONS = {
    "sensory": (
        "LOC_X",
        "LOC_Y",  # location in world
        "BOUNDARY_DIST_X",
        "BOUNDARY_DIST",
        "BOUNDARY_DIST_Y",  # distance to nearest edge
        "GENETIC_SIM_FWD",  # genetic similarity forward
        "LAST_MOVE_DIR_X",
        "LAST_MOVE_DIR_Y",  # +- amount of movement in last movement
        "LONGPROBE_POP_FWD",
        "LONGPROBE_BAR_FWD",  # long look for pop/bar forward
        "POPULATION",
        "POPULATION_FWD",
        "POPULATION_LR",  # population density
        "OSC0",  # oscilator +- value
        "AGE",
        "BARRIER_FWD",
        "BARRIER_LR",  # neighborhood barrier distance
        "RANDOM",
    ),
    "internal": tuple("N" + str(i) for i in range(MAX_INTERNAL_NEURON)),
    "action": (
        "MOVE_X",
        "MOVE_Y",
        "MOVE_FORWARD",
        "MOVE_RL",
        "MOVE_RANDOM",
        "SET_OSCILLATOR_PERIOD",
        # SET_RESPONSIVENESS,
        # "KILL_FORWARD"
    ),
}
# dimensions of output image of world
IMAGE_WIDTH, IMAGE_HEIGHT = WORLD_WIDTH * 8, WORLD_HEIGHT * 8


def survival_criteria(x: int, y: int) -> bool:
    """x and y are the creature's position in world, return true if it survives"""
    return (y < 20 or y > 80) and (x < 20 or x > 80)  # survives if in corner
