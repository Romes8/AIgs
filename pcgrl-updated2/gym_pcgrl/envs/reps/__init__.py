from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation

# Define the relevant representations for Sokoban
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation
}