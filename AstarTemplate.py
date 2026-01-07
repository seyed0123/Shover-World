import numpy as np
import heapq
from collections import defaultdict, deque
import itertools

EMPTY = 0
BARRIER = 100
LAVA = -100
BOX_MIN = 1
BOX_MAX = 10

DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
BARRIER_MAKER = 5
HELLIFY = 6

INITIAL_FORCE = 40.0
UNIT_FORCE = 10.0
R_LAVA = 40.0
R_BARRIER = 10.0


class Agent:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.plan = []
        self.previous_action = None
        self.previous_pos = None
        self.recent_states = deque(maxlen=200)

    def reset(self, obs):
        self.plan.clear()
        self.previous_action = None
        self.previous_pos = None
        self.recent_states.clear()

    def act(self, obs):
        grid = obs["grid"]
        stamina = obs["stamina"]

        key = self._state_key(grid)
        self.recent_states.append(key)

        if self.plan:
            return self.plan.pop(0)

        if self._must_create_hellify(grid):
            squares = self._find_perfect_squares(grid)
            if any(n >= 3 for n, _ in squares):
                return ((0, 0), HELLIFY)

            plan = self._a_star_plan(grid, stamina, hellify_only=True)
            if plan:
                self.plan = plan[1:]
                return plan[0]

        plan = self._a_star_plan(grid, stamina)
        if plan:
            self.plan = plan[1:]
            return plan[0]

        action = self._greedy_best_action(grid)
        if action:
            return action

        return ((0, 0), 1)

    def _has_lava(self, grid):
        return np.any(grid == LAVA)

    def _must_create_hellify(self, grid):
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

    def _a_star_plan(self, grid, stamina, hellify_only=False):
        start_key = self._state_key(grid)
        open_set = []
        counter = itertools.count()

        heapq.heappush(open_set, (0, 0, next(counter), grid, []))
        g_cost = defaultdict(lambda: np.inf)
        g_cost[start_key] = 0
        visited = set()

        initial_boxes = np.sum((grid >= BOX_MIN) & (grid <= BOX_MAX))
        expansions = 0

        while open_set:
            expansions += 1
            if expansions > 300000:
                return []

            _, cost, _, current, path = heapq.heappop(open_set)
            key = self._state_key(current)
            if key in visited:
                continue
            visited.add(key)

            squares = self._find_perfect_squares(current)
            if hellify_only:
                if any(n >= 3 for n, _ in squares):
                    return path
            else:
                boxes = np.sum((current >= BOX_MIN) & (current <= BOX_MAX))
                if boxes < initial_boxes:
                    return path

            actions = self._get_valid_push_actions(current)
            if not hellify_only:
                actions.append(((0, 0), BARRIER_MAKER))
            actions.append(((0, 0), HELLIFY))

            for action in actions:
                new_grid, delta = self._apply_action(current, action)
                if new_grid is None:
                    continue

                new_key = self._state_key(new_grid)
                new_cost = cost - delta
                if new_cost >= g_cost[new_key]:
                    continue

                g_cost[new_key] = new_cost
                h = self._heuristic(new_grid)
                heapq.heappush(
                    open_set,
                    (new_cost + h, new_cost, next(counter), new_grid, path + [action])
                )

        return []

    def _heuristic(self, grid):
        boxes = np.sum((grid >= BOX_MIN) & (grid <= BOX_MAX))
        if boxes == 0:
            return 0

        dist = self._compute_dist_to_sink(grid)
        score = 0
        for r, c in np.argwhere((grid >= BOX_MIN) & (grid <= BOX_MAX)):
            score += dist[r, c] if dist[r, c] < np.inf else 20

        squares = self._find_perfect_squares(grid)
        if any(n >= 3 for n, _ in squares):
            score -= 100

        if not self._has_lava(grid):
            score += 50

        return score + boxes * 5

    def _apply_action(self, grid, action):
        (r, c), act = action
        new_grid = grid.copy()
        delta = -1.0

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
                    destroyed += 1
                elif new_grid[nr, nc] == LAVA:
                    destroyed += 1
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

    def _state_key(self, grid):
        g = grid.copy()
        g[(g >= BOX_MIN) & (g <= BOX_MAX)] = 1
        return g.tobytes()

    def _simulate_push(self, grid, r, c, delta):
        dr, dc = delta
        chain = []
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and BOX_MIN <= grid[r, c] <= BOX_MAX:
            chain.append((r, c))
            r += dr
            c += dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] not in [EMPTY, LAVA]:
            return []
        return chain

    def _get_valid_push_actions(self, grid):
        actions = []
        for r, c in np.argwhere((grid >= BOX_MIN) & (grid <= BOX_MAX)):
            for d in DIRS:
                if self._simulate_push(grid, r, c, DIRS[d]):
                    actions.append(((r, c), d))
        return actions

    def _find_perfect_squares(self, grid):
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

    def _is_isolated(self, grid, r, c, n):
        for rr in range(max(0, r-1), min(grid.shape[0], r+n+1)):
            for cc in range(max(0, c-1), min(grid.shape[1], c+n+1)):
                if r <= rr < r+n and c <= cc < c+n:
                    continue
                if BOX_MIN <= grid[rr, cc] <= BOX_MAX:
                    return False
        return True

    def _compute_dist_to_sink(self, grid):
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

    def _greedy_best_action(self, grid):
        best = None
        best_score = np.inf
        for action in self._get_valid_push_actions(grid):
            new_grid, delta = self._apply_action(grid, action)
            if new_grid is None:
                continue
            score = -delta + self._heuristic(new_grid)
            if score < best_score:
                best_score = score
                best = action
        return best
