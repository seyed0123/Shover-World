import heapq
import numpy as np
import logging
from datetime import datetime

BOX = 10
LAVA = -100
BARRIER = 100
EMPTY = 0
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
BARRIER_MAKER = 4
HELLIFY = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class AStarSolver:
    def __init__(self, env, max_expansions=150_000, log_interval=2000):
        self.env = env
        self.n_rows = env.n_rows
        self.n_cols = env.n_cols
        self.max_expansions = max_expansions
        self.log_interval = log_interval
        
        # Static lava cells
        self.lava_cells = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.env.grid[r, c] == LAVA:
                    self.lava_cells.append((r, c))

    def _heuristic(self):
        grid = self.env.grid
        box_locs = np.argwhere(grid == BOX)
        num_boxes = len(box_locs)
        if num_boxes == 0: return 0


        return num_boxes * 20.0

    def solve(self):
        start = datetime.now()
        snap0 = self.env._snapshot()
        init_stamina = snap0["stamina"]

        open_set, visited = [], {}
        counter = expansions = 0
        best_h = float("inf")

        h0 = self._heuristic()
        heapq.heappush(open_set, (h0, 0, counter, [], snap0))

        while open_set and expansions < self.max_expansions:
            f, g, _, path, snap = heapq.heappop(open_set)
            key = snap["grid"].tobytes()

            if key in visited and visited[key] >= snap["stamina"]:
                continue
            visited[key] = snap["stamina"]

            self.env._restore(snap)

            if not np.any(self.env.grid == BOX):
                logger.info(
                    f"SOLVED! Time: {(datetime.now()-start).total_seconds():.2f}s | "
                    f"Steps: {len(path)} | Expansions: {expansions} | "
                    f"Final Stamina: {self.env.stamina:.0f}"
                )
                return path

            expansions += 1
            if expansions % self.log_interval == 0:
                logger.info(
                    f"Exp: {expansions} | Boxes: {np.sum(self.env.grid == BOX)} | "
                    f"Stamina: {self.env.stamina:.0f} | H: {self._heuristic():.1f}"
                )

            for action in self.env._get_valid_actions():
                self.env._restore(snap)
                _, _, _, _, info = self.env.step(action)
                if not info["valid_action"]:
                    continue

                g_new = (init_stamina - self.env.stamina) / 10.0
                h_new = self._heuristic()
                best_h = min(best_h, h_new)

                counter += 1
                heapq.heappush(
                    open_set,
                    (g_new + h_new, g_new, counter, path + [action], self.env._snapshot())
                )

        logger.warning(f"Failed. Expansions: {expansions} | Best H: {best_h}")
        return []

