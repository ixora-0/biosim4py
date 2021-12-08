import random
from pyvis.network import Network
from constants import *
from PIL import Image, ImageColor, ImageDraw


def random_genome() -> str:
    """Create random genome"""
    return " ".join(
        "%06x" % random.randrange(16 ** GENE_LENGTH) for _ in range(GENOME_LENGTH)
    )


def is_in_world(x: int, y: int) -> bool:
    """Return true if the given coordinates are within the boundaries of the world"""
    return (0 <= y < WORLD_HEIGHT) and (0 <= x < WORLD_WIDTH)


def get_brain_visualization(neurons: dict[str, dict]) -> Network:
    """Return a pyvis Network of the neurons"""
    net = Network(bgcolor="#222222", font_color="#dddddd", directed=True)
    net.toggle_physics(True)
    net.set_edge_smooth("dynamic")

    for t in neurons:
        for n in neurons[t]:
            net.add_node(n, color=NEURON_COLORS[t])
    for t in ["sensory", "internal"]:
        for n in neurons[t]:
            for cc in neurons[t][n].connected_to:
                x = int((cc["weight"] + MAX_WEIGHT) * 255 / (MAX_WEIGHT * 2))
                net.add_edge(
                    n,
                    cc["neuron"].name,
                    color=("#%02x%02x%02x" % (255 - x, x, 0)),
                    width=4,
                )
    net.set_options(
        """
        var options = {
          "physics": {
            "barnesHut": {
              "centralGravity": 0.2,
              "springLength": 250,
              "springConstant": 0.03,
              "damping": 0.5
            },
            "maxVelocity": 25,
            "minVelocity": 0.75
          }
        }
    """
    )

    return net


def draw_world(world) -> Image:
    """Return an image of the world"""
    im = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    gridx = IMAGE_WIDTH // WORLD_WIDTH
    gridy = IMAGE_HEIGHT // WORLD_HEIGHT
    draw = ImageDraw.Draw(im)
    for y in range(WORLD_HEIGHT):
        for x in range(WORLD_WIDTH):
            if world[y][x] != 0:
                draw.rectangle(
                    (gridx * x, gridy * y, gridx * (x + 1), gridy * (y + 1)),
                    fill=ImageColor.getcolor(world[y][x].color, "RGB"),
                )
    return im
