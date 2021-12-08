import random
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np
from bitstring import BitArray

from constants import *
import utils


class Creature:
    def __init__(self, genome, x=None, y=None):
        self.genome = genome
        self.x = x
        self.y = y
        self.facing_x = 0
        self.facing_y = 0
        self.osc_period = OSC_INITIAL_PERIOD

        # set random direction on initialization
        while self.facing_x == 0 and self.facing_y == 0:
            self.facing_x = random.randint(-1, 1)
            self.facing_y = random.randint(-1, 1)

        # color is the first and last three hex characters of genome
        self.color = "#" + genome[:3] + genome[-3:]

        # decoding genome (mRNA)
        action_neurons = set()  # a set of action neurons in the brain, no duplicates
        connections = []  # list of dicts, indicating source, target and weight
        for gene in genome.split(" "):
            bits = bin(int(gene, 16))[2:].zfill(GENE_LENGTH * 4)

            # decode source neuron
            if bits[0] == "1":
                source_type = "internal"
            else:
                source_type = "sensory"
            source_id = BitArray(bin=bits[1 : NUM_ID_BITS + 1]).uint % len(
                NEURONS[source_type]
            )

            # decode sink neuron
            if bits[NUM_ID_BITS + 1] == "1":
                sink_type = "action"
            else:
                sink_type = "internal"
            sink_id = BitArray(
                bin=bits[NUM_ID_BITS + 2 : 2 * NUM_ID_BITS + 2]
            ).uint % len(NEURONS[sink_type])

            # decode weights
            weight = BitArray(bin=bits[2 * NUM_ID_BITS + 2 :]).int / WEIGHT_SCALE
            if sink_type == "action":
                action_neurons.add(NEURONS[sink_type][sink_id])
            # add to the connections list
            connections.append(
                {
                    "from_type": source_type,
                    "from": NEURONS[source_type][source_id],
                    "to_type": sink_type,
                    "to": NEURONS[sink_type][sink_id],
                    "weight": weight,
                }
            )

        # creating the neuron  (rRNA)
        self.neurons = {"sensory": {}, "internal": {}, "action": {}}  # the "brain"
        for action in action_neurons:
            # Create each action neurons first, then the neurons
            # that are connected to each action neuron, then the neurons
            # that are connected to those, etc.
            # This ensures that all neurons that are in anyway
            # connected to action neurons are created, and ignores those which aren't

            self.neurons["action"][action] = ActionNeuron(self, action)
            children = [self.neurons["action"][action]]

            while len(children) > 0:
                new_children = []
                for child in children:
                    cc = {"neuron": child, "weight": 0}
                    for c in connections:
                        if c["to"] == child.name:
                            cc["weight"] = c["weight"]
                            if c["from"] in self.neurons[c["from_type"]]:
                                # if the neuron is already created, just add the connection
                                self.neurons[c["from_type"]][
                                    c["from"]
                                ].connected_to.append(cc)
                            else:
                                # create the neuron
                                if c["from_type"] == "internal":
                                    new_neuron = InternalNeuron(self, c["from"])
                                    new_children.append(
                                        new_neuron
                                    )  # internal neurons can be a child of other neurons, sensory neurons can't
                                else:
                                    new_neuron = SensoryNeuron(self, c["from"])
                                self.neurons[c["from_type"]][c["from"]] = new_neuron
                                self.neurons[c["from_type"]][
                                    c["from"]
                                ].connected_to.append(cc)
                            connections.remove(c)
                children = new_children

    def step_brain(self):
        # calculate the output of action neurons
        for t in self.neurons:
            for n in self.neurons[t]:
                self.neurons[t][n].evaluate()
                self.neurons[t][n].feed_inputs()

    def execute_actions(self):
        # from action neurons' outputs, do the corresponding action
        dx, dy = 0, 0
        # probability to move in x, y direction, if negative than the move is in the negative direction
        for n in self.neurons["action"]:
            output = self.neurons["action"][n].output  # this is the raw sum, not scaled
            match n:
                case "MOVE_X":
                    dx += output
                case "MOVE_Y":
                    dy += output
                case "MOVE_FORWARD":
                    dx += self.facing_x
                    dy += self.facing_y
                case "MOVE_RL":
                    dx -= self.facing_y
                    dy += self.facing_x
                case "MOVE_RANDOM":
                    if output > 0:
                        # only move randomly if theaction neuron is activated
                        dx += random.random() * 2 - 1
                        dy += random.random() * 2 - 1
                case "SET_OSCILLATOR_PERIOD":
                    self.osc_period = 2.5 + math.exp(3 * (math.tanh(output) + 1))
                case _:
                    raise TypeError(
                        f"Invalid action neuron {self.neurons['action'][n].name}"
                    )

        dx, dy = math.tanh(dx), math.tanh(dy)  # turn into a probability (from 0 to 1)
        probX, probY = random.random() < abs(dx), random.random() < abs(dy)
        signX, signY = -1 if dx < 0 else 1, -1 if dy < 0 else 1
        self.move(probX * signX, probY * signY)

    def move(self, dx, dy):
        # move the creature in the world
        global world
        tx, ty = self.x + dx, self.y + dy
        if utils.is_in_world(tx, ty) and world[ty][tx] == 0:
            world[self.y][self.x] = 0
            world[ty][tx] = self
            self.x += dx
            self.y += dy
        if (dx, dy) == (0, 0):
            self.facing_x = dx
            self.facing_y = dy


class Neuron:
    """Parent data structure for neurons"""

    def __init__(self, creature: Creature, name: str):
        self.connected_to = []  # {neuron that is connected to: weight}
        self.creature = creature  # owner of neurons
        self.name = name
        self.inputs = {}  # {source neuron name: value of input}
        self.output = 0

    def feed_inputs(self):
        """feed output to all connected neurons"""
        for c in self.connected_to:
            c["neuron"].inputs[self.name] = self.output * c["weight"]

    def evaluate(self):
        self.output = 0


class SensoryNeuron(Neuron):
    def evaluate(self):
        """Calulate the output for the sensor neuron"""
        match self.name:
            case "LOC_X":
                # location on the x axis
                self.output = self.creature.x / WORLD_WIDTH
            case "LOC_Y":
                # location on the y axis
                self.output = self.creature.y / WORLD_HEIGHT
            case "BOUNDARY_DIST_X":
                # Distance to the nearest boundary on the x axis
                self.output = min(
                    self.creature.x, WORLD_WIDTH - self.creature.x - 1
                ) / int(WORLD_WIDTH / 2 - 1)
            case "BOUNDARY_DIST_Y":
                # Distance to the nearest boundary on the x axis
                self.output = min(
                    self.creature.y, WORLD_HEIGHT - self.creature.y - 1
                ) / int(WORLD_HEIGHT / 2 - 1)
            case "BOUNDARY_DIST":
                # Distance to the nearest boundary
                self.output = (
                    2
                    * min(
                        self.creature.x,
                        WORLD_WIDTH - self.creature.x - 1,
                        self.creature.y,
                        WORLD_HEIGHT - self.creature.y - 1,
                    )
                    / int(max(WORLD_WIDTH / 2 - 1, WORLD_HEIGHT / 2 - 1))
                )
            case "GENETIC_SIM_FWD":
                # How genetically similar the creature directly forward is (return 0 if noo one's there)
                tx, ty = (
                    self.creature.x + self.creature.facing_x,
                    self.creature.y + self.creature.facing_y,
                )
                if utils.is_in_world(tx, ty) and type(world[ty][tx]) == Creature:
                    # returns how many bits are different, scaled to (0, 1)
                    g1 = "".join(
                        [
                            bin(int(gene, 16))[2:].zfill(24)
                            for gene in self.creature.genome.split(" ")
                        ]
                    )
                    g2 = "".join(
                        [
                            bin(int(gene, 16))[2:].zfill(24)
                            for gene in world[ty][tx].genome.split(" ")
                        ]
                    )

                    N = 0
                    L = len(g1)
                    for i in range(L):
                        if g1[i] != g2[i]:
                            N += 1
                    self.output = 1 - min(1, (2 * N) / L)
            case "LAST_MOVE_DIR_X":
                # Direction on the x axis
                self.output = (self.creature.facing_x + 1) / 2
            case "LAST_MOVE_DIR_Y":
                # Direction on the y axis
                self.output = (self.creature.facing_y + 1) / 2
            case "LONGPROBE_POP_FWD":
                # How far away a creature is in the forward direction
                for d in range(1, LONG_PROBE_DISTANCE):
                    tx = self.creature.x + d * self.creature.facing_x
                    ty = self.creature.y + d * self.creature.facing_y
                    if utils.is_in_world(tx, ty) and type(world[ty][tx]) == Creature:
                        self.output = (
                            LONG_PROBE_DISTANCE - d + 1
                        ) / LONG_PROBE_DISTANCE
                        break
                self.output = 0
            case "LONGPROBE_BAR_FWD":
                # How far away a barrier is in the forward direction
                for d in range(1, LONG_PROBE_DISTANCE):
                    tx = self.creature.x + d * self.creature.facing_x
                    ty = self.creature.y + d * self.creature.facing_y
                    if utils.is_in_world(tx, ty) and world[ty][tx] == "B":
                        self.output = (
                            LONG_PROBE_DISTANCE - d + 1
                        ) / LONG_PROBE_DISTANCE
                        break
                self.output = 0
            case "POPULATION":
                # Population desnsity in surrounding area
                c = 0
                for dx in range(
                    -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                ):
                    for dy in range(
                        -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                    ):
                        if dx == 0 and dy == 0:
                            continue
                        tx = self.creature.x + dx
                        ty = self.creature.y + dy
                        if (
                            utils.is_in_world(tx, ty)
                            and type(world[ty][tx]) == Creature
                        ):
                            c += 1
                self.output = c / ((POPULATION_SENSOR_RADIUS * 2 + 1) ** 2 - 1)
            case "POPULATION_FWD":
                # From https://github.com/davidrmiller/biosim4/blob/main/src/getSensor.cpp:
                # Converts the population along the specified axis to the sensor range. The
                # locations of neighbors are scaled by the inverse of their distance times
                # the positive absolute cosine of the difference of their angle and the
                # specified axis. The maximum positive or negative magnitude of the sum is
                # about 2*radius. We don't adjust for being close to a border, so populations
                # along borders and in corners are commonly sparser than away from borders.
                # An empty neighborhood results in a sensor value exactly midrange; below
                # midrange if the population density is greatest in the reverse direction,
                # above midrange if density is greatest in forward direction.
                c = 0
                for dx in range(
                    -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                ):
                    for dy in range(
                        -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                    ):
                        if dx == 0 and dy == 0:
                            continue
                        tx = self.creature.x + dx
                        ty = self.creature.y + dy
                        if (
                            utils.is_in_world(tx, ty)
                            and type(world[ty][tx]) == Creature
                        ):
                            c += (
                                self.creature.facing_x * dx
                                + self.creature.facing_y * dy
                            ) / (dx * dx + dy * dy)
                maxsum = 3 * (2 * POPULATION_SENSOR_RADIUS + 1)
                assert c >= -maxsum and c <= maxsum
                self.output = (c / maxsum + 1) / 2
            case "POPULATION_LR":
                # Similar to the POPULATION_FWD neuron, but for left-right direction
                c = 0
                for dx in range(
                    -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                ):
                    for dy in range(
                        -POPULATION_SENSOR_RADIUS, POPULATION_SENSOR_RADIUS + 1
                    ):
                        if dx == 0 and dy == 0:
                            continue
                        tx = -self.creature.y + dx
                        ty = self.creature.x + dy
                        if (
                            utils.is_in_world(tx, ty)
                            and type(world[ty][tx]) == Creature
                        ):
                            c += (
                                -self.creature.facing_y * dx
                                + self.creature.facing_x * dy
                            ) / (dx * dx + dy * dy)
                maxsum = 3 * (2 * POPULATION_SENSOR_RADIUS + 1)
                assert c >= -maxsum and c <= maxsum
                self.output = (c / maxsum + 1) / 2
            case "OSC0":
                # Output of a cos function with phase determined by creature's age
                phase = (sim_step % self.creature.osc_period) / self.creature.osc_period
                self.output = (math.cos(phase * 2 * math.pi) + 1) / 2
                self.output = (
                    1 if self.output > 1 else (0 if self.output < 0 else self.output)
                )
            case "AGE":
                # Age of the creature, which is the same as age of the world
                self.output = sim_step / STEP_PER_GENERATION
            case "BARRIER_FWD":
                # How far away a barrier is in the forward direction
                self.output = self.short_probe_barrier(
                    self.creature.facing_x, self.creature.facing_y
                )
            case "BARRIER_LR":
                # How far away a barrier is in the left-right direction
                self.output = self.short_probe_barrier(
                    -self.creature.facing_y, self.creature.facing_x
                )
            case "RANDOM":
                # random float in (0, 1), uniform
                self.output = random.random()
            case _:
                raise TypeError(f"Invalid sensory neuron {self.name}")

    def short_probe_barrier(self, x_axis: int, y_axis: int):
        """From https://github.com/davidrmiller/biosim4/blob/main/src/getSensor.cpp:
        Converts the number of locations (not including loc) to the next barrier location
        along opposite directions of the specified axis to the sensor range. If no barriers
        are found, the result is sensor mid-range. Ignores agents in the path."""

        count_fwd = 0
        count_rev = 0
        num_locs_to_test = SHORT_PROBE_DISTANCE

        tx, ty = self.creature.x + x_axis, self.creature.y + y_axis
        while (
            num_locs_to_test > 0 and utils.is_in_world(tx, ty) and world[ty][tx] != "B"
        ):
            count_fwd += 1
            tx += x_axis
            ty += y_axis
            num_locs_to_test -= 1
        if num_locs_to_test > 0 and not utils.is_in_world(tx, ty):
            count_fwd = num_locs_to_test

        num_locs_to_test = SHORT_PROBE_DISTANCE
        tx, ty = self.creature.x - x_axis, self.creature.y - y_axis
        while (
            num_locs_to_test > 0 and utils.is_in_world(tx, ty) and world[ty][tx] != "B"
        ):
            count_rev += 1
            tx -= x_axis
            ty -= y_axis
            num_locs_to_test -= 1
        if num_locs_to_test > 0 and not utils.is_in_world(tx, ty):
            count_rev = num_locs_to_test
        return (count_fwd - count_rev + SHORT_PROBE_DISTANCE) / (
            2 * SHORT_PROBE_DISTANCE
        )


class InternalNeuron(Neuron):
    def evaluate(self):
        """Calulate the output for the the internal neuron"""
        self.output = math.tanh(sum(self.inputs.values()))


class ActionNeuron(Neuron):
    def evaluate(self):
        """Calulate the output for the the action neuron (raw sum)"""
        self.output = sum(self.inputs.values())


def initialize_world():
    """Empty world"""
    global world
    global sim_step
    world = [[0 for _ in range(WORLD_WIDTH)] for _ in range(WORLD_HEIGHT)]
    sim_step = 0


def populate_world(population: list[Creature] = []):
    """Populate the world with population, if empty then populate with creatures that has random genome"""
    global world
    if len(population) == 0:
        for _ in range(POPULATION):
            x, y = random.randrange(WORLD_WIDTH), random.randrange(WORLD_HEIGHT)
            while world[y][x] != 0:
                x, y = random.randrange(WORLD_WIDTH), random.randrange(WORLD_HEIGHT)
            world[y][x] = Creature(utils.random_genome(), x, y)
    else:
        for creature in population:
            x, y = random.randrange(WORLD_WIDTH), random.randrange(WORLD_HEIGHT)
            while world[y][x] != 0:
                x, y = random.randrange(WORLD_WIDTH), random.randrange(WORLD_HEIGHT)
            world[y][x] = creature
            creature.x = x
            creature.y = y


def get_population() -> list[Creature]:
    """Return the list of creatures in the world"""
    return [
        world[y][x]
        for y in range(WORLD_HEIGHT)
        for x in range(WORLD_WIDTH)
        if type(world[y][x]) == Creature
    ]


def step_world():
    """Calculate action neurons' outputs of all creatures, then execute their actions"""
    global sim_step
    for y in range(WORLD_HEIGHT):
        for x in range(WORLD_WIDTH):
            if type(world[y][x]) == Creature:
                world[y][x].step_brain()
                world[y][x].execute_actions()
    sim_step += 1


def apply_survival_criteria():
    """Remove all creatures that doesn't meet the survial critera"""
    global world
    for y in range(WORLD_HEIGHT):
        for x in range(WORLD_WIDTH):
            if type(world[y][x]) == Creature:
                if not survival_criteria(x, y):
                    world[y][x] = 0


def repopulate():
    """From the current population, select 2 random parents,
    create a new genome from the parents' using 1 point crossover,
    repeat until a new the generation is filled"""
    population = get_population()
    next_gen = []
    for _ in range(POPULATION):
        parents = [
            bin(int(p.genome.replace(" ", ""), 16))[2:].zfill(
                GENOME_LENGTH * GENE_LENGTH * 4
            )
            for p in random.sample(population, k=2)
        ]
        crossover_point = random.randrange(1, GENOME_LENGTH * GENE_LENGTH * 4)

        child = parents[0][:crossover_point] + parents[1][crossover_point:]
        child = "".join(
            [
                str(int(bool(int(b)) != (random.random() < POINT_MUTATION_RATE)))
                for b in child
            ]
        )
        child = [
            child[i : i + GENE_LENGTH * 4]
            for i in range(0, GENOME_LENGTH * GENE_LENGTH * 4, GENE_LENGTH * 4)
        ]
        child = " ".join([hex(int(i, 2))[2:].zfill(GENE_LENGTH) for i in child])

        next_gen.append(Creature(child))

    initialize_world()
    populate_world(next_gen)


def record_generation(path: str = "videos/gen.mp4"):
    """Simulate a new generation, then save it's video in path"""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(path, fourcc, 30, (IMAGE_WIDTH, IMAGE_HEIGHT))

    for i in range(STEP_PER_GENERATION):
        print(
            f"Recording simulation: {str(i).ljust(3)}/{STEP_PER_GENERATION}", end="\r"
        )
        video.write(cv2.cvtColor(np.array(utils.draw_world()), cv2.COLOR_RGB2BGR))
        step_world()

    print("\nOutputing video...")
    video.release()


sim_step = 0
world = [[0 for _ in range(WORLD_WIDTH)] for _ in range(WORLD_HEIGHT)]


if __name__ == "__main__":
    for gen in range(NUM_GENERATIONS):
        initialize_world()
        populate_world()
        for i in range(STEP_PER_GENERATION):
            print(f"\rGENERATION {str(gen).ljust(4)}: {str(i).ljust(4)}", end="")
            step_world()

    record_generation()
