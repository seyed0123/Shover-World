import heapq
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Tuple, Union
from environment import ShoverWorldEnv, BOX, LAVA, BARRIER, EMPTY, UP, RIGHT, DOWN, LEFT, BARRIER_MAKER, HELLIFY


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class AStarSolver:
    """
    An AI solver that uses the A* search algorithm to find the optimal path to the goal in the ShoverWorld environment.
    logs progress at regular intervals.
    and limits the number of state expansions to avoid excessive computation.
    """
    def __init__(self, env: ShoverWorldEnv, max_expansions=150_000, log_interval=2000):
        self.env = env
        self.n_rows = env.n_rows
        self.n_cols = env.n_cols
        self.max_expansions = max_expansions
        self.log_interval = log_interval
        # TODO: another precomputations if needed

    def solve(self)->List[Tuple[Union[np.int64, int], int]]:
        """
        Solve the environment using A* search.
        returns a list of actions to reach the goal.
        the actions are in the format of (position_indx, action_type) where:
        position_indx (numpy int64) = row * number_of_cols + col
        action_type is one of these Actions:
            UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
            BARRIER_MAKER = 4
            HELLIFY = 5
        """
        logger.info("=== A* Search ===")
        raise NotImplementedError("Implement the A* search algorithm.")
