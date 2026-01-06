import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Any
import pygame
from collections import defaultdict
import os

BOX = 10
LAVA = -100
BARRIER = 100
EMPTY = 0

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
BARRIER_MAKER = 4
HELLIFY = 5

class ShoverWorldEnv(gym.Env):
    """
    Shover-World: A box-selection-based grid environment with stamina mechanics.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # Cell type constants
    LAVA = LAVA
    EMPTY = EMPTY   
    BOX = BOX
    BARRIER = BARRIER

    # Action constants
    UP = UP
    RIGHT = RIGHT
    DOWN = DOWN
    LEFT = LEFT
    BARRIER_MAKER = BARRIER_MAKER
    HELLIFY = HELLIFY
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        n_rows: int = 8,
        n_cols: int = 8,
        max_timestep: int = 400,
        number_of_boxes: int = 10,
        number_of_barriers: int = 3,
        number_of_lavas: int = 2,
        initial_stamina: float = 1000.0,
        initial_force: float = 40.0,
        unit_force: float = 10.0,
        perf_sq_initial_age: int = 50,
        map_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.map_path = map_path
        self.initial_grid = None

        # --- DYNAMIC MAP LOADING ---
        # If a map path is provided, load it first to determine dimensions
        if self.map_path and os.path.exists(self.map_path):
            try:
                self.initial_grid = self._load_map(self.map_path)
                # Override dimensions with loaded map size
                n_rows, n_cols = self.initial_grid.shape
                print(f"Loaded map from {self.map_path}: {n_rows}x{n_cols}")
            except Exception as e:
                print(f"Error loading map {self.map_path}: {e}")
                # Fallback to default dimensions if loading fails
                self.initial_grid = None
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        
        # Rest of initialization using the (potentially updated) dimensions
        self.max_timestep = max_timestep
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas
        self.initial_stamina = initial_stamina
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.perf_sq_initial_age = perf_sq_initial_age
        self.render_mode = render_mode
        
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()
        
        # Action space depends on n_rows * n_cols
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.n_rows * self.n_cols),
            spaces.Discrete(6)
        ))
        
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100, high=100, shape=(self.n_rows, self.n_cols), dtype=np.int32),
            "stamina": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "previous_selected_position": spaces.Box(low=-1, high=max(self.n_rows, self.n_cols), shape=(2,), dtype=np.int32),
            "previous_action": spaces.Discrete(7)
        })
        
        # Initialize state variables
        self.grid = None
        self.stamina = None
        self.timestep = None
        self.previous_selected_position = None
        self.previous_action = None
        
        self.box_stationary = defaultdict(lambda: {
            'up': True, 'down': True, 'left': True, 'right': True
        })
        
        self.perfect_squares = []
        self.boxes_destroyed = 0
        self.window = None
        self.clock = None
        self.cell_size = 60
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Use pre-loaded grid if available, otherwise generate new random map
        if self.initial_grid is not None:
            self.grid = self.initial_grid.copy()
        else:
            self.grid = self._generate_random_map()
        
        # Ensure borders are lava (important if map file didn't include them)
        # You can comment this out if you trust your map files to have correct borders
        self._add_lava_borders()
        
        self.stamina = self.initial_stamina
        self.timestep = 0
        self.previous_selected_position = np.array([-1, -1])
        self.previous_action = 0
        self.boxes_destroyed = 0
        
        self.box_stationary.clear()
        
        # Reset all box stationary status
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] == self.BOX:
                    # Accessing it creates the default entry
                    _ = self.box_stationary[(r, c)]
        
        self.perfect_squares = []
        self._detect_perfect_squares()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        position_idx, action_type = action
        action_type += 1  # Convert from 0-5 to 1-6
        
        # Convert position index to (row, col)
        row = position_idx // self.n_cols
        col = position_idx % self.n_cols
        
        self.timestep += 1
        reward = 0.0
        valid_action = False
        chain_length = 0
        initial_force_charged = False
        lava_destroyed_this_step = 0
        
        # Store previous action info
        self.previous_selected_position = np.array([row, col])
        self.previous_action = action_type
        
        # Process action
        if action_type in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            # Movement action
            result = self._process_movement(row, col, action_type)
            valid_action = result['valid']
            chain_length = result['chain_length']
            initial_force_charged = result['initial_force_charged']
            lava_destroyed_this_step = result['lava_destroyed']
            reward += result['reward']
            
            if not valid_action:
                # Invalid action penalty: 1/10 of box movement stamina
                penalty = self.unit_force / 10.0
                self.stamina -= penalty
                reward -= penalty
        
        elif action_type == self.BARRIER_MAKER:
            result = self._process_barrier_maker()
            valid_action = result['valid']
            reward += result['reward']
            
            if not valid_action:
                penalty = self.unit_force / 10.0
                self.stamina -= penalty
                reward -= penalty
        
        elif action_type == self.HELLIFY:
            result = self._process_hellify()
            valid_action = result['valid']
            lava_destroyed_this_step = result['lava_destroyed']
            reward += result['reward']
            
            if not valid_action:
                penalty = self.unit_force / 10.0
                self.stamina -= penalty
                reward -= penalty
        
        # Update perfect squares
        self._detect_perfect_squares()
        self._age_perfect_squares()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.timestep >= self.max_timestep
        
        info = self._get_info()
        info.update({
            'valid_action': valid_action,
            'chain_length': chain_length,
            'initial_force_charged': initial_force_charged,
            'lava_destroyed_this_step': lava_destroyed_this_step,
        })
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _process_movement(self, row: int, col: int, direction: int) -> Dict:
        """Process a movement action (push boxes)."""
        result = {
            'valid': False,
            'chain_length': 0,
            'initial_force_charged': False,
            'lava_destroyed': 0,
            'reward': 0.0
        }
        
        # Check if selected position has a box
        if self.grid[row, col] != self.BOX:
            return result
        
        # Get direction vector
        dir_map = {
            self.UP: (-1, 0, 'up'),
            self.DOWN: (1, 0, 'down'),
            self.LEFT: (0, -1, 'left'),
            self.RIGHT: (0, 1, 'right')
        }
        dr, dc, dir_name = dir_map[direction]
        
        # Find chain of boxes
        chain_positions = []
        r, c = row, col
        while 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.grid[r, c] == self.BOX:
            chain_positions.append((r, c))
            r += dr
            c += dc
        
        # Check destination of chain
        next_r, next_c = r, c
        
        # If out of bounds or barrier, invalid
        if not (0 <= next_r < self.n_rows and 0 <= next_c < self.n_cols):
            return result
        
        if self.grid[next_r, next_c] == self.BARRIER:
            return result
        
        # If destination is empty or lava, we can push
        if self.grid[next_r, next_c] in [self.EMPTY, self.LAVA]:
            result['valid'] = True
            result['chain_length'] = len(chain_positions)
            
            # Calculate stamina cost
            head_box = chain_positions[0]
            # Check if box is stationary in this direction
            is_stationary = self.box_stationary[head_box][dir_name]
            
            cost = self.unit_force * len(chain_positions)
            if is_stationary:
                cost += self.initial_force
                result['initial_force_charged'] = True
            
            self.stamina -= cost
            reward = -cost
            
            # Move boxes in reverse order to avoid conflicts
            boxes_to_lava = 0
            new_positions = []
            
            for r_box, c_box in reversed(chain_positions):
                new_r, new_c = r_box + dr, c_box + dc
                
                # Remove box from old position first
                self.grid[r_box, c_box] = self.EMPTY
                
                # Check if moving into lava
                if self.grid[new_r, new_c] == self.LAVA:
                    boxes_to_lava += 1
                    # Box is destroyed, don't place it
                else:
                    # Place box in new position
                    self.grid[new_r, new_c] = self.BOX
                    new_positions.append((new_r, new_c))
                
                # Remove old position from stationary tracking
                if (r_box, c_box) in self.box_stationary:
                    del self.box_stationary[(r_box, c_box)]
            
            # Update stationary status for moved boxes
            for pos in new_positions:
                # Mark as non-stationary in the direction of movement
                self.box_stationary[pos][dir_name] = False
                # Keep other directions as stationary
                for other_dir in ['up', 'down', 'left', 'right']:
                    if other_dir != dir_name and other_dir not in self.box_stationary[pos]:
                        self.box_stationary[pos][other_dir] = True
            
            # Lava refund
            if boxes_to_lava > 0:
                refund = self.initial_force * boxes_to_lava
                self.stamina += refund
                result['reward'] = reward + refund
                result['lava_destroyed'] = boxes_to_lava
                self.boxes_destroyed += boxes_to_lava
            else:
                result['reward'] = reward
        
        return result
    
    def _process_barrier_maker(self) -> Dict:
        """Process Barrier Maker special action."""
        result = {'valid': False, 'reward': 0.0}
        
        # Find perfect squares with n >= 2
        valid_squares = [sq for sq in self.perfect_squares if sq[0] >= 2]
        
        if not valid_squares:
            return result
        
        # Get oldest square
        oldest = min(valid_squares, key=lambda x: x[3])
        size, top, left, _ = oldest
        
        # Convert all boxes to barriers
        for r in range(top, top + size):
            for c in range(left, left + size):
                if self.grid[r, c] == self.BOX:
                    self.grid[r, c] = self.BARRIER
                    # Remove from stationary tracking
                    if (r, c) in self.box_stationary:
                        del self.box_stationary[(r, c)]
        
        # Award stamina
        stamina_bonus = self.unit_force * (size * size)
        self.stamina += stamina_bonus
        result['reward'] = stamina_bonus
        result['valid'] = True
        
        # Remove from perfect squares list
        self.perfect_squares.remove(oldest)
        
        return result
    
    def _process_hellify(self) -> Dict:
        """Process Hellify special action."""
        result = {'valid': False, 'lava_destroyed': 0, 'reward': 0.0}
        
        # Find perfect squares with n > 2
        valid_squares = [sq for sq in self.perfect_squares if sq[0] > 2]
        
        if not valid_squares:
            return result
        
        # Get oldest square
        oldest = min(valid_squares, key=lambda x: x[3])
        size, top, left, _ = oldest
        
        boxes_destroyed = 0
        
        # Erase boundary boxes, convert interior to lava
        for r in range(top, top + size):
            for c in range(left, left + size):
                if r == top or r == top + size - 1 or c == left or c == left + size - 1:
                    # Boundary - erase
                    if self.grid[r, c] == self.BOX:
                        self.grid[r, c] = self.EMPTY
                        boxes_destroyed += 1
                        # Remove from stationary tracking
                        if (r, c) in self.box_stationary:
                            del self.box_stationary[(r, c)]
                else:
                    # Interior - convert to lava
                    if self.grid[r, c] == self.BOX:
                        self.grid[r, c] = self.LAVA
                        boxes_destroyed += 1
                        # Remove from stationary tracking
                        if (r, c) in self.box_stationary:
                            del self.box_stationary[(r, c)]
        
        result['valid'] = True
        result['lava_destroyed'] = boxes_destroyed
        self.boxes_destroyed += boxes_destroyed
        
        # Reward for destroyed boxes
        result['reward'] = self.initial_force * boxes_destroyed
        
        # Remove from perfect squares list
        self.perfect_squares.remove(oldest)
        
        return result
    
    def _detect_perfect_squares(self):
        """Detect all perfect squares on the map."""
        new_squares = []
        
        # Try all possible square sizes and positions
        for size in range(2, min(self.n_rows, self.n_cols) + 1):
            for top in range(self.n_rows - size + 1):
                for left in range(self.n_cols - size + 1):
                    if self._is_perfect_square(top, left, size):
                        # Check if already tracked (maintain timestep)
                        found = False
                        for sq in self.perfect_squares:
                            if sq[0] == size and sq[1] == top and sq[2] == left:
                                new_squares.append(sq)
                                found = True
                                break
                        
                        if not found:
                            new_squares.append((size, top, left, self.timestep))
        
        self.perfect_squares = new_squares
    
    def _is_perfect_square(self, top: int, left: int, size: int) -> bool:
        """Check if region forms a perfect square."""
        # All cells in the n×n region must be boxes
        for r in range(top, top + size):
            for c in range(left, left + size):
                if self.grid[r, c] != self.BOX:
                    return False
        
        # No boxes adjacent to perimeter (8-neighborhood)
        for r in range(top - 1, top + size + 1):
            for c in range(left - 1, left + size + 1):
                # Skip interior
                if top <= r < top + size and left <= c < left + size:
                    continue
                
                # Check boundary
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    if self.grid[r, c] == self.BOX:
                        return False
        
        return True
    
    def _age_perfect_squares(self):
        """Age perfect squares and dissolve old ones."""
        to_remove = []
        
        for i, (size, top, left, creation_time) in enumerate(self.perfect_squares):
            age = self.timestep - creation_time
            
            if age >= self.perf_sq_initial_age:
                # Dissolve: all boxes become empty
                for r in range(top, top + size):
                    for c in range(left, left + size):
                        if self.grid[r, c] == self.BOX:
                            self.grid[r, c] = self.EMPTY
                            # Remove from stationary tracking
                            if (r, c) in self.box_stationary:
                                del self.box_stationary[(r, c)]
                
                to_remove.append(i)
        
        # Remove dissolved squares
        for i in reversed(to_remove):
            self.perfect_squares.pop(i)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # No boxes remain
        if np.sum(self.grid == self.BOX) == 0:
            return True
        
        # Stamina depleted
        if self.stamina <= 0:
            return True
        
        return False
    
    def _generate_random_map(self) -> np.ndarray:
        """Generate a random map with boxes, barriers, and lava."""
        # Start with empty grid (excluding borders for now)
        grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        
        # Place boxes randomly (avoid borders which will be lava)
        available_positions = [(r, c) for r in range(1, self.n_rows - 1) 
                               for c in range(1, self.n_cols - 1)]
        
        box_positions = self.np_random.choice(len(available_positions), 
                                              size=min(self.number_of_boxes, len(available_positions)), 
                                              replace=False)
        for idx in box_positions:
            r, c = available_positions[idx]
            grid[r, c] = self.BOX
        
        # Remove used positions
        available_positions = [pos for i, pos in enumerate(available_positions) 
                               if i not in box_positions]
        
        # Place barriers
        if len(available_positions) >= self.number_of_barriers:
            barrier_positions = self.np_random.choice(len(available_positions), 
                                                      size=self.number_of_barriers, 
                                                      replace=False)
            for idx in barrier_positions:
                r, c = available_positions[idx]
                grid[r, c] = self.BARRIER
            
            available_positions = [pos for i, pos in enumerate(available_positions) 
                                   if i not in barrier_positions]
        
        # Place lava cells inside map
        if len(available_positions) >= self.number_of_lavas:
            lava_positions = self.np_random.choice(len(available_positions), 
                                                   size=self.number_of_lavas, 
                                                   replace=False)
            for idx in lava_positions:
                r, c = available_positions[idx]
                grid[r, c] = self.LAVA
        
        return grid
    
    def _add_lava_borders(self):
        """Add lava borders around the entire map."""
        # Top and bottom rows
        self.grid[0, :] = self.LAVA
        self.grid[-1, :] = self.LAVA
        
        # Left and right columns
        self.grid[:, 0] = self.LAVA
        self.grid[:, -1] = self.LAVA
    
    def _load_map(self, path: str) -> np.ndarray:
        """Load map from file."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        try:
            # Try parsing as integers
            return np.array([[int(x) for x in l.split()] for l in lines], dtype=np.int32)
        except ValueError:
            # Try symbolic format
            symbol_map = {'.': 0, 'B': 10, '#': 100, 'L': -100, 'A': 0}
            grid = []
            for line in lines:
                row = [symbol_map.get(c, 0) for c in line]
                grid.append(row)
            return np.array(grid, dtype=np.int32)
    
    def _get_obs(self) -> Dict:
        """Get current observation."""
        return {
            "grid": self.grid.copy(),
            "stamina": np.array([self.stamina], dtype=np.float32),
            "previous_selected_position": self.previous_selected_position.copy(),
            "previous_action": self.previous_action
        }
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            "timestep": self.timestep,
            "stamina": self.stamina,
            "number_of_boxes": int(np.sum(self.grid == self.BOX)),
            "boxes_destroyed": self.boxes_destroyed,
            "perfect_squares": [(size, top, left) for size, top, left, _ in self.perfect_squares]
        }
    
    def render(self):
        """Render the environment using Pygame."""
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            
            window_width = self.n_cols * self.cell_size
            window_height = self.n_rows * self.cell_size + 60
            
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Shover-World")
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((255, 255, 255))
        
        # Define colors
        colors = {
            self.LAVA: (255, 0, 0),
            self.EMPTY: (240, 240, 240),
            self.BOX: (139, 69, 19),
            self.BARRIER: (50, 50, 50)
        }
        
        # Draw grid
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell_value = self.grid[r, c]
                color = colors.get(cell_value, (255, 255, 255))
                
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)
        
        # Draw selected position indicator
        if self.previous_selected_position[0] >= 0:
            r, c = self.previous_selected_position
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                rect = pygame.Rect(
                    c * self.cell_size + 5,
                    r * self.cell_size + 5,
                    self.cell_size - 10,
                    self.cell_size - 10
                )
                pygame.draw.rect(self.window, (0, 255, 0), rect, 3)
        
        # Draw HUD
        hud_y = self.n_rows * self.cell_size
        pygame.draw.rect(self.window, (220, 220, 220), 
                        (0, hud_y, self.n_cols * self.cell_size, 60))
        
        font = pygame.font.Font(None, 24)
        
        info = self._get_info()
        hud_text = f"Timestep: {info['timestep']} | Stamina: {self.stamina:.1f} | Boxes: {info['number_of_boxes']} | Destroyed: {info['boxes_destroyed']}"
        
        text_surface = font.render(hud_text, True, (0, 0, 0))
        self.window.blit(text_surface, (10, hud_y + 10))
        
        # Perfect squares info
        if self.perfect_squares:
            ps_text = f"Perfect Squares: {len(self.perfect_squares)}"
            ps_surface = font.render(ps_text, True, (0, 100, 0))
            self.window.blit(ps_surface, (10, hud_y + 35))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _snapshot(self):
        """Create a snapshot of the current state."""
        return {
            "grid": self.grid.copy(),
            "stamina": float(self.stamina),
            "timestep": int(self.timestep),
            "boxes_destroyed": int(self.boxes_destroyed),
            "box_stationary": {k: v.copy() for k, v in self.box_stationary.items()},
            "perfect_squares": list(self.perfect_squares),
        }
    def _restore(self, snapshot: Dict):
        """Restore state from snapshot."""
        self.grid = snapshot["grid"].copy()
        self.stamina = snapshot["stamina"]
        self.timestep = snapshot["timestep"]
        self.boxes_destroyed = snapshot["boxes_destroyed"]
        self.box_stationary = defaultdict(
            lambda: {'up': True, 'down': True, 'left': True, 'right': True},
            {k: v.copy() for k, v in snapshot["box_stationary"].items()}
        )
        self.perfect_squares = list(snapshot["perfect_squares"])
    
    def _get_valid_actions(self):
        """
        dont forget to call env.reset() before using this function
        Returns a list of valid actions in the current state.
        """
        actions = []
        
        squares = self.perfect_squares
        if any(s[0] >= 2 for s in squares):
            actions.append((0, self.BARRIER_MAKER))
        if any(s[0] > 2 for s in squares):
            actions.append((0, self.HELLIFY))

        # Move Actions
        box_rows, box_cols = np.where(self.grid == self.BOX)
        dirs = [(-1,0), (0,1), (1,0), (0,-1)]
        
        for r, c in zip(box_rows, box_cols):
            pos_idx = r * self.n_cols + c
            
            for action_type, (dr, dc) in enumerate(dirs):
                curr_r, curr_c = r + dr, c + dc
                valid = False
                # Chain Check loop
                while 0 <= curr_r < self.n_rows and 0 <= curr_c < self.n_cols:
                    cell = self.grid[curr_r, curr_c]
                    if cell == self.EMPTY or cell == self.LAVA:
                        valid = True
                        break
                    elif cell == self.BOX:
                        curr_r += dr
                        curr_c += dc
                    else: # Barrier
                        break
                if valid:
                    actions.append((pos_idx, action_type))
        return actions