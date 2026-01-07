
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import heapq
import numpy as np


@dataclass
class _Snapshot:
    grid: np.ndarray
    stamina: float
    timestep: int
    boxes_destroyed: int
    box_stationary: Dict
    perfect_squares: List


class AStarSolver:
    """
    This class matches main.py:
        solver = AStarSolver(env, max_expansions=..., log_interval=...)
        plan = solver.solve()

    plan MUST be a list of actions, where each action is:
        (position_idx, action_type)

    position_idx = row * env.n_cols + col
    action_type  = env.UP-1 / env.RIGHT-1 / env.DOWN-1 / env.LEFT-1
    """

    def __init__(self, env, max_expansions: int = 50_000, log_interval: int = 500):
        self.env = env
        self.max_expansions = int(max_expansions)
        self.log_interval = int(log_interval)

        try:
            self._dir_action_types = [
                self.env.UP - 1,
                self.env.RIGHT - 1,
                self.env.DOWN - 1,
                self.env.LEFT - 1,
            ]
        except Exception:
            self._dir_action_types = [0, 1, 2, 3]

        # For heuristic: detect constants if present
        self._BOX = getattr(self.env, "BOX", 1)
        self._LAVA = getattr(self.env, "LAVA", -2)


    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            grid=self.env.grid.copy(),
            stamina=float(getattr(self.env, "stamina", 0.0)),
            timestep=int(getattr(self.env, "timestep", 0)),
            boxes_destroyed=int(getattr(self.env, "boxes_destroyed", 0)),
            box_stationary=dict(getattr(self.env, "box_stationary", {})),
            perfect_squares=list(getattr(self.env, "perfect_squares", [])),
        )

    def _restore(self, s: _Snapshot) -> None:
        self.env.grid = s.grid.copy()
        if hasattr(self.env, "stamina"):
            self.env.stamina = s.stamina
        if hasattr(self.env, "timestep"):
            self.env.timestep = s.timestep
        if hasattr(self.env, "boxes_destroyed"):
            self.env.boxes_destroyed = s.boxes_destroyed
        if hasattr(self.env, "box_stationary"):
            self.env.box_stationary.clear()
            self.env.box_stationary.update(s.box_stationary)
        if hasattr(self.env, "perfect_squares"):
            self.env.perfect_squares = list(s.perfect_squares)


    def _key(self, s: _Snapshot) -> Tuple[bytes, int, int]:
        # include grid + coarse stamina + timestep to reduce collisions
        stamina_bucket = int(round(s.stamina))
        return (s.grid.astype(np.int16, copy=False).tobytes(), stamina_bucket, s.timestep)


    def _boxes_positions(self, grid: np.ndarray) -> List[Tuple[int, int]]:

        if np.issubdtype(grid.dtype, np.integer):
            mask = (grid == self._BOX) if self._BOX is not None else (grid > 0)
        else:
            mask = (grid == self._BOX)
        coords = np.argwhere(mask)
        return [tuple(map(int, x)) for x in coords]

    def _lava_positions(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        coords = np.argwhere(grid == self._LAVA)
        return [tuple(map(int, x)) for x in coords]

    def _is_goal(self, s: _Snapshot) -> bool:

        grid_boxes = len(self._boxes_positions(s.grid))
        if grid_boxes == 0:
            return True

        n_boxes = getattr(self.env, "number_of_boxes", None)
        if n_boxes is not None and s.boxes_destroyed >= int(n_boxes):
            return True
        return False

    def _heuristic(self, s: _Snapshot) -> float:

        boxes = self._boxes_positions(s.grid)
        if not boxes:
            return 0.0
        lavas = self._lava_positions(s.grid)
        if not lavas:
            return float(len(boxes) * 50)

        dist_sum = 0
        for (br, bc) in boxes:
            best = min(abs(br - lr) + abs(bc - lc) for (lr, lc) in lavas)
            dist_sum += best
        return float(5 * len(boxes) + dist_sum)


    def _try_action_from_current(self, action: Tuple[int, int]) -> Optional[_Snapshot]:

        prev = self._snapshot()
        try:
            _obs, _reward, _terminated, _truncated, info = self.env.step(action)
        except Exception:

            self._restore(prev)
            return None

        valid = True
        if isinstance(info, dict) and "valid_action" in info:
            valid = bool(info["valid_action"])

        nxt = self._snapshot()
        self._restore(prev)

        return nxt if valid else None


    def solve(self) -> List[Tuple[int, int]]:
        start = self._snapshot()
        start_key = self._key(start)


        open_heap: List[Tuple[float, int, Tuple[bytes, int, int]]] = []
        heapq.heappush(open_heap, (self._heuristic(start), 0, start_key))

        g_score: Dict[Tuple[bytes, int, int], int] = {start_key: 0}
        came_from: Dict[Tuple[bytes, int, int], Tuple[Tuple[bytes, int, int], Tuple[int, int]]] = {}
        snapshots: Dict[Tuple[bytes, int, int], _Snapshot] = {start_key: start}
        closed: set[Tuple[bytes, int, int]] = set()

        expansions = 0

        while open_heap and expansions < self.max_expansions:
            f, g, key = heapq.heappop(open_heap)
            if key in closed:
                continue
            closed.add(key)

            cur = snapshots[key]

            self._restore(cur)

            if self._is_goal(cur):

                plan_rev: List[Tuple[int, int]] = []
                k = key
                while k != start_key:
                    pk, act = came_from[k]
                    plan_rev.append(act)
                    k = pk
                plan_rev.reverse()
                return plan_rev

            expansions += 1
            if self.log_interval and expansions % self.log_interval == 0:
                boxes_left = len(self._boxes_positions(cur.grid))
                print(f"[A*] expanded={expansions} open={len(open_heap)} g={g} boxes_left={boxes_left}")


            boxes = self._boxes_positions(cur.grid)
            for (r, c) in boxes:
                pos_idx = r * int(self.env.n_cols) + c
                for action_type in self._dir_action_types:
                    action = (pos_idx, int(action_type))
                    nxt = self._try_action_from_current(action)
                    if nxt is None:
                        continue

                    nk = self._key(nxt)
                    ng = g + 1
                    if ng < g_score.get(nk, 10**9):
                        g_score[nk] = ng
                        came_from[nk] = (key, action)
                        snapshots[nk] = nxt
                        nf = ng + self._heuristic(nxt)
                        heapq.heappush(open_heap, (nf, ng, nk))

        
        return []
