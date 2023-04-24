import numpy as np
from PyFlyt.core import Aviary
from PyFlytMenagerie import Revtriplane

if __name__ == "__main__":
    # the starting position and orientations
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])

    # define a new drone type
    drone_type_mappings = dict()
    drone_type_mappings["revtriplane"] = Revtriplane

    # environment setup
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=True,
        drone_type_mappings=drone_type_mappings,
        drone_type="revtriplane",
    )

    # simulate for 1000 steps (1000/120 ~= 8 seconds)
    for i in range(1000):
        env.step()
