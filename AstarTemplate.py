import heapq
import numpy as np
import logging
import itertools
from collections import defaultdict, deque
from typing import List, Tuple, Union
from environment import ShoverWorldEnv, LAVA, BARRIER, EMPTY, UP, RIGHT, DOWN, LEFT, BARRIER_MAKER, HELLIFY

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

EMPTY = 0
BARRIER = 100
LAVA = -100
BOX_MIN = 1
BOX_MAX = 10

DIRS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
INITIAL_FORCE = 40.0
UNIT_FORCE = 10.0
R_LAVA = 40.0
R_BARRIER = 10.0

class AStarSolver:
    def __init__(self, env: ShoverWorldEnv, max_expansions=300_000, log_interval=5000):
        self.env = env
        self.n_rows = env.n_rows
        self.n_cols = env.n_cols
        self.max_expansions = max_expansions
        self.log_interval = log_interval

    def solve(self) -> List[Tuple[Union[np.int64, int], int]]:
        logger.info("=== A* Search Started ===")
        grid = self.env.grid.copy()
        stamina = self.env.stamina

        if self._must_create_hellify(grid):
            squares = self._find_perfect_squares(grid)
            if any(n >= 3 for n, _ in squares):
                logger.info("Direct HELLIFY action found.")
                return [(0, HELLIFY)]

            plan = self._a_star_plan(grid, stamina, hellify_only=True)
            if plan:
                logger.info(f"Hellify plan found with {len(plan)} steps.")
                return plan

        plan = self._a_star_plan(grid, stamina)
        if plan:
            logger.info(f"Full plan found with {len(plan)} steps.")
            return plan

        action = self._greedy_best_action(grid, stamina)
        if action:
            logger.info("Greedy action selected as fallback.")
            return [action]

        logger.warning("No action found; returning default.")
        return []

    def _state_key(self, grid: np.ndarray, stamina: float) -> bytes:
        g = grid.copy()
        g[(g >= BOX_MIN) & (g <= BOX_MAX)] = 1
        return g.tobytes() + bytes([int(stamina / 10)])  # Simple stamina bucketing to reduce states

    def _has_lava(self, grid: np.ndarray) -> bool:
        return np.any(grid == LAVA)

    def _must_create_hellify(self, grid: np.ndarray) -> bool:
        if self._has_lava(grid):
            return False
        box_count = np.sum((grid >= BOX_MIN) & (grid <= BOX_MAX))
        if box_count < 9:
            return False

        dist = self._compute_dist_to_sink(grid)
        for r, c in np.argwhere((grid >= BOX_MIN) & (grid <= BOX_MAX)):
            if dist[r, c] == 0:
                return False

        return True

    def _a_star_plan(self, grid: np.ndarray, stamina: float, hellify_only: bool = False) -> List[Tuple[int, int]]:
        start_key = self._state_key(grid, stamina)
        open_set = []
        counter = itertools.count()
        heapq.heappush(open_set, (0, 0, next(counter), grid.copy(), stamina, []))
        g_cost = defaultdict(lambda: float('inf'))
        g_cost[start_key] = 0
        visited = set()

        initial_boxes = np.sum((grid >= BOX_MIN) & (grid <= BOX_MAX))
        expansions = 0

        while open_set:
            expansions += 1
            if expansions % self.log_interval == 0:
                logger.info(f"Expansions: {expansions}, Open set: {len(open_set)}, Best cost: {open_set[0][1]}")
            if expansions > self.max_expansions:
                logger.warning(f"Max expansions ({self.max_expansions}) reached. No plan found.")
                return []

            _, cost, _, current_grid, current_stamina, path = heapq.heappop(open_set)
            key = self._state_key(current_grid, current_stamina)
            if key in visited:
                continue
            visited.add(key)

            squares = self._find_perfect_squares(current_grid)
            if hellify_only:
                if any(n >= 3 for n, _ in squares):
                    logger.info(f"Hellify goal reached after {expansions} expansions.")
                    return path
            else:
                boxes = np.sum((current_grid >= BOX_MIN) & (current_grid <= BOX_MAX))
                if boxes < initial_boxes:
                    logger.info(f"Goal reached: {initial_boxes - boxes} boxes destroyed after {expansions} expansions.")
                    return path

            actions = self._get_valid_push_actions(current_grid)
            if not hellify_only:
                if self._find_perfect_squares(current_grid):  # Only if squares exist
                    actions.append((0, BARRIER_MAKER))
            if any(n >= 3 for n, _ in squares):
                actions.append((0, HELLIFY))

            for action in actions:
                new_grid, delta = self._apply_action(current_grid, action)
                if new_grid is None:
                    continue

                new_stamina = current_stamina + delta
                if new_stamina <= 0:
                    continue

                new_key = self._state_key(new_grid, new_stamina)
                new_cost = cost + max(0, -delta)  # Adjust cost to minimize effort, reward positive delta
                if new_cost >= g_cost[new_key]:
                    continue

                g_cost[new_key] = new_cost
                h = self._heuristic(new_grid)
                f = new_cost + h
                heapq.heappush(
                    open_set,
                    (f, new_cost, next(counter), new_grid.copy(), new_stamina, path + [action])
                )

        logger.warning("Search exhausted. No plan found.")
        return []

    def _heuristic(self, grid: np.ndarray) -> float:
        boxes = np.sum((grid >= BOX_MIN) & (grid <= BOX_MAX))
        if boxes == 0:
            return 0

        dist = self._compute_dist_to_sink(grid)
        score = 0.0
        for r, c in np.argwhere((grid >= BOX_MIN) & (grid <= BOX_MAX)):
            d = dist[r, c]
            score += d if d < np.inf else 50  # Higher penalty for trapped boxes

        squares = self._find_perfect_squares(grid)
        max_square = max((n for n, _ in squares), default=0)
        score -= 20 * max_square  # Encourage larger squares

        if not self._has_lava(grid) and max_square >= 3:
            score -= 100  # Big incentive for hellify potential

        return score + boxes * 10  # Stronger weight on reducing boxes

    def _apply_action(self, grid: np.ndarray, action: Tuple[int, int]) -> Tuple[Union[np.ndarray, None], float]:
        position_idx, act = action
        r, c = divmod(position_idx, self.n_cols) if position_idx > 0 else (0, 0)
        new_grid = grid.copy()
        delta = 0.0  # No base -1; adjust per action

        if act in DIRS:
            chain = self._simulate_push(new_grid, r, c, DIRS[act])
            if not chain:
                return None, delta

            destroyed = 0
            for cr, cc in reversed(chain):
                nr, nc = cr + DIRS[act][0], cc + DIRS[act][1]
                val = new_grid[cr, cc]
                new_grid[cr, cc] = EMPTY

                if not (0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]):
                    destroyed += val  # Value-based destroy reward?
                elif new_grid[nr, nc] == LAVA:
                    destroyed += val
                else:
                    new_grid[nr, nc] = val

            delta -= INITIAL_FORCE + UNIT_FORCE * len(chain)
            delta += R_LAVA * destroyed

        elif act == BARRIER_MAKER:
            squares = self._find_perfect_squares(new_grid)
            if not squares:
                return None, delta
            n, (tr, tc) = max(squares, key=lambda x: x[0])
            new_grid[tr:tr + n, tc:tc + n] = BARRIER
            delta += R_BARRIER * n * n

        elif act == HELLIFY:
            squares = [s for s in self._find_perfect_squares(new_grid) if s[0] >= 3]
            if not squares:
                return None, delta
            n, (tr, tc) = max(squares, key=lambda x: x[0])
            delta += R_LAVA * (n - 2) ** 2
            for r0 in range(tr, tr + n):
                for c0 in range(tc, tc + n):
                    if r0 in (tr, tr + n - 1) or c0 in (tc, tc + n - 1):
                        new_grid[r0, c0] = EMPTY
                    else:
                        new_grid[r0, c0] = LAVA

        return new_grid, delta

    def _simulate_push(self, grid: np.ndarray, r: int, c: int, delta: Tuple[int, int]) -> List[Tuple[int, int]]:
        dr, dc = delta
        chain = []
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and BOX_MIN <= grid[r, c] <= BOX_MAX:
            chain.append((r, c))
            r += dr
            c += dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] not in [EMPTY, LAVA]:
            return []
        return chain

    def _get_valid_push_actions(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        actions = []
        for r, c in np.argwhere((grid >= BOX_MIN) & (grid <= BOX_MAX)):
            for d in DIRS:
                if self._simulate_push(grid, r, c, DIRS[d]):
                    position_idx = r * self.n_cols + c
                    actions.append((position_idx, d))
        return actions

    def _find_perfect_squares(self, grid: np.ndarray) -> List[Tuple[int, Tuple[int, int]]]:
        rows, cols = grid.shape
        squares = []
        for r in range(rows):
            for c in range(cols):
                max_n = min(rows - r, cols - c)
                for n in range(2, max_n + 1):
                    sub = grid[r:r+n, c:c+n]
                    if not np.all((sub >= BOX_MIN) & (sub <= BOX_MAX)):
                        break
                    if self._is_isolated(grid, r, c, n):
                        squares.append((n, (r, c)))
        return squares

    def _is_isolated(self, grid: np.ndarray, r: int, c: int, n: int) -> bool:
        for rr in range(max(0, r-1), min(grid.shape[0], r + n + 1)):
            for cc in range(max(0, c-1), min(grid.shape[1], c + n + 1)):
                if r <= rr < r + n and c <= cc < c + n:
                    continue
                if BOX_MIN <= grid[rr, cc] <= BOX_MAX:
                    return False
        return True

    def _compute_dist_to_sink(self, grid: np.ndarray) -> np.ndarray:
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)
        q = deque()

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == LAVA or r in (0, rows-1) or c in (0, cols-1):
                    if grid[r, c] != BARRIER:
                        dist[r, c] = 0
                        q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != BARRIER:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist

    def _greedy_best_action(self, grid: np.ndarray, stamina: float) -> Tuple[int, int]:
        best = None
        best_score = float('inf')
        actions = self._get_valid_push_actions(grid) + [(0, BARRIER_MAKER), (0, HELLIFY)]
        for action in actions:
            new_grid, delta = self._apply_action(grid, action)
            if new_grid is None or stamina + delta <= 0:
                continue
            score = max(0, -delta) + self._heuristic(new_grid)  # Minimize cost + h
            if score < best_score:
                best_score = score
                best = action
        return best