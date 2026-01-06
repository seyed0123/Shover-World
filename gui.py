import pygame
import numpy as np
import os
from environment import ShoverWorldEnv
from typing import Optional, Tuple


class EnhancedShoverWorldGUI:

    """
    Enhanced Interactive GUI for Shover-World.

    Features:
    - Dark mode theme
    - Texture-based graphics
    - Resizable window
    - Mouse and keyboard controls
    """
    
    def __init__(self, env: ShoverWorldEnv, initial_cell_size: int = 50, fullscreen: bool = False):
        self.env = env
        self.cell_size = initial_cell_size
        self.min_cell_size = 30
        self.max_cell_size = 100
        
        # Initialize Pygame
        pygame.init()
        
        # Get display info for sizing
        display_info = pygame.display.Info()
        self.screen_width = display_info.current_w
        self.screen_height = display_info.current_h
        
        # Window dimensions
        self.sidebar_width = 280
        self.hud_height = 100
        
        # Calculate initial window size to fit screen
        self._calculate_window_size()
        
        # Create resizable window
        if fullscreen:
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height), 
                                                  pygame.FULLSCREEN | pygame.SCALED)
        else:
            self.window = pygame.display.set_mode((self.window_width, self.window_height), 
                                                  pygame.RESIZABLE)
        
        pygame.display.set_caption("Shover-World - Enhanced Edition")
        self.clock = pygame.time.Clock()
        
        # Load textures
        self.textures = {}
        self._load_textures()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
        # Dark mode colors
        self.colors = {
            'background': (20, 20, 25),
            'grid_bg': (30, 30, 35),
            'sidebar': (25, 25, 30),
            'hud': (28, 28, 32),
            'border': (50, 50, 55),
            'text': (220, 220, 220),
            'text_dim': (150, 150, 155),
            'text_highlight': (255, 100, 100),
            'accent': (100, 180, 255),
            'success': (100, 255, 150),
            'warning': (255, 200, 100),
            'danger': (255, 100, 100),
            'selected': (100, 255, 100),
            'hover': (255, 255, 100),
            'perfect_square': (100, 200, 255),
            'panel_dark': (18, 18, 22),
            'button': (60, 120, 200),
            'button_hover': (80, 140, 220),
        }
        
        # Game state
        self.selected_position = None
        self.hover_position = None
        self.last_action_info = None
        self.message = ""
        self.message_timer = 0
        self.message_color = self.colors['text']
        
        # Animation
        self.animation_cells = []
        self.moving_boxes = {}  # Track opacity for moving boxes: (r, c) -> {'opacity': float, 'duration': int, 'max_duration': int}
        self.transformation_effects = {}  # Track transformation animations: (r, c) -> {'type': str, 'opacity': float, 'duration': int}
        
        # Zoom controls
        self.show_help = False
        
    def _calculate_window_size(self):
        """Calculate window size to fit on screen."""
        # Maximum grid size that fits on screen
        max_grid_width = int(self.screen_width * 0.6)
        max_grid_height = int(self.screen_height * 0.75)
        
        # Calculate cell size that fits
        max_cell_from_width = max_grid_width // self.env.n_cols
        max_cell_from_height = max_grid_height // self.env.n_rows
        
        # Use smaller to ensure everything fits
        self.cell_size = min(self.cell_size, max_cell_from_width, max_cell_from_height)
        self.cell_size = max(self.cell_size, self.min_cell_size)
        
        # Calculate window dimensions
        self.grid_width = self.env.n_cols * self.cell_size
        self.grid_height = self.env.n_rows * self.cell_size
        self.window_width = self.grid_width + self.sidebar_width
        self.window_height = self.grid_height + self.hud_height
    
    def _load_textures(self):
        """
        Load texture images for game elements.
        
        Expected images in 'assets' folder:
        - lava.png: 64x64px, red/orange lava texture
        - box.png: 64x64px, wooden box texture
        - barrier.png: 64x64px, stone/metal barrier texture
        - floor.png: 64x64px, floor tile texture
        - selected.png: 64x64px, selection overlay (transparent with green border)
        - hover.png: 64x64px, hover overlay (transparent with yellow tint)
        """
        
        asset_path = "assets"
        texture_files = {
            'lava': 'lava.png',
            'box': 'box.png',
            'barrier': 'barrier.png',
            'floor': 'floor.png',
            'selected': 'selected.png',
            'hover': 'hover.png',
        }
        
        # Create default textures if files don't exist
        for key, filename in texture_files.items():
            filepath = os.path.join(asset_path, filename)
            
            if os.path.exists(filepath):
                try:
                    texture = pygame.image.load(filepath).convert_alpha()
                    self.textures[key] = texture
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
                    self.textures[key] = self._create_default_texture(key)
            else:
                self.textures[key] = self._create_default_texture(key)
    
    def _create_default_texture(self, texture_type: str) -> pygame.Surface:
        """Create a default colored texture when image is missing."""
        size = 64
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        if texture_type == 'lava':
            # Red gradient
            for y in range(size):
                intensity = 200 + int(55 * (y / size))
                pygame.draw.rect(surface, (intensity, 50, 30), (0, y, size, 1))
        
        elif texture_type == 'box':
            # Brown with wood grain effect
            surface.fill((139, 90, 43))
            for i in range(0, size, 8):
                pygame.draw.line(surface, (120, 75, 35), (i, 0), (i, size), 2)
        
        elif texture_type == 'barrier':
            # Dark gray stone
            surface.fill((50, 50, 55))
            # Add brick lines
            for y in range(0, size, 16):
                pygame.draw.line(surface, (40, 40, 45), (0, y), (size, y), 2)
            for x in range(0, size, 16):
                pygame.draw.line(surface, (40, 40, 45), (x, 0), (x, size), 2)
        
        elif texture_type == 'floor':
            # Dark tile pattern
            surface.fill((40, 40, 45))
            pygame.draw.rect(surface, (35, 35, 40), (2, 2, size-4, size-4))
        
        elif texture_type == 'selected':
            # Transparent with green border
            surface.fill((0, 0, 0, 0))
            pygame.draw.rect(surface, (100, 255, 100, 200), (0, 0, size, size), 4)
        
        elif texture_type == 'hover':
            # Semi-transparent yellow overlay
            surface.fill((255, 255, 100, 80))
        
        return surface
    
    def _draw_texture(self, surface: pygame.Surface, texture_key: str, rect: pygame.Rect):
        """Draw a scaled texture to fit the rect."""
        if texture_key in self.textures:
            scaled_texture = pygame.transform.scale(self.textures[texture_key], 
                                                    (rect.width, rect.height))
            surface.blit(scaled_texture, rect)
    
    def run(self):
        """Main game loop."""
        obs, info = self.env.reset()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self._handle_mouse_click(event.pos)
                    elif event.button == 4:  # Scroll up (zoom in)
                        self._zoom(1)
                    elif event.button == 5:  # Scroll down (zoom out)
                        self._zoom(-1)
                
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    terminated, truncated = self._handle_keyboard(event.key)
                    if terminated or truncated:
                        self._show_message("Episode ended! Press R to reset", 
                                         self.colors['text_highlight'])
            
            # Update animations
            self._update_animations()
            
            # Render
            self._render()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        self.env.close()
        pygame.quit()
    
    def _handle_resize(self, width: int, height: int):
        """Handle window resize."""
        self.window_width = width
        self.window_height = height
        
        # Recalculate grid dimensions
        available_width = width - self.sidebar_width
        available_height = height - self.hud_height
        
        # Calculate new cell size
        new_cell_width = available_width // self.env.n_cols
        new_cell_height = available_height // self.env.n_rows
        
        self.cell_size = min(new_cell_width, new_cell_height)
        self.cell_size = max(self.min_cell_size, min(self.cell_size, self.max_cell_size))
        
        self.grid_width = self.env.n_cols * self.cell_size
        self.grid_height = self.env.n_rows * self.cell_size
    
    def _zoom(self, direction: int):
        """Zoom in or out."""
        old_size = self.cell_size
        self.cell_size += direction * 5
        self.cell_size = max(self.min_cell_size, min(self.cell_size, self.max_cell_size))
        
        if old_size != self.cell_size:
            self.grid_width = self.env.n_cols * self.cell_size
            self.grid_height = self.env.n_rows * self.cell_size
            self.window_width = self.grid_width + self.sidebar_width
            self.window_height = self.grid_height + self.hud_height
            
            # Resize window
            self.window = pygame.display.set_mode((self.window_width, self.window_height), 
                                                  pygame.RESIZABLE)
    
    def _handle_mouse_click(self, pos: Tuple[int, int]):
        """Handle mouse click events."""
        x, y = pos
        
        # Check if click is on grid
        if x < self.grid_width and y < self.grid_height:
            col = x // self.cell_size
            row = y // self.cell_size
            
            if 0 <= row < self.env.n_rows and 0 <= col < self.env.n_cols:
                if self.env.grid[row, col] == self.env.BOX:
                    self.selected_position = (row, col)
                    self._show_message(f"Selected box at ({row}, {col})", self.colors['success'])
                else:
                    self.selected_position = None
                    self._show_message("No box at this position", self.colors['warning'])
    
    def _handle_mouse_motion(self, pos: Tuple[int, int]):
        """Handle mouse motion for hover effects."""
        x, y = pos
        
        if x < self.grid_width and y < self.grid_height:
            col = x // self.cell_size
            row = y // self.cell_size
            
            if 0 <= row < self.env.n_rows and 0 <= col < self.env.n_cols:
                self.hover_position = (row, col)
            else:
                self.hover_position = None
        else:
            self.hover_position = None
    
    def _handle_keyboard(self, key: int) -> Tuple[bool, bool]:
        """Handle keyboard events."""
        terminated = False
        truncated = False
        
        # Reset
        if key == pygame.K_r:
            obs, info = self.env.reset()
            self.selected_position = None
            self._show_message("Environment reset!", self.colors['success'])
            return False, False
        
        # Quit
        if key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return False, False
        
        # Toggle help
        if key == pygame.K_F1:
            self.show_help = not self.show_help
            return False, False
        
        # Fullscreen toggle
        if key == pygame.K_F11:
            pygame.display.toggle_fullscreen()
            return False, False
        
        # Random action
        if key == pygame.K_SPACE:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_action_info = info
            self._show_message(f"Random action: Reward={reward:.1f}", self.colors['text'])
            self._add_animation_effect(info)
            return terminated, truncated
        
        # Actions requiring selected box
        if self.selected_position is None and key not in [pygame.K_h, pygame.K_b]:
            self._show_message("Please select a box first!", self.colors['warning'])
            return False, False
        
        # Determine action
        action_type = None
        action_name = ""
        
        if key == pygame.K_UP or key == pygame.K_w:
            action_type = self.env.UP - 1
            action_name = "UP"
        elif key == pygame.K_DOWN or key == pygame.K_s:
            action_type = self.env.DOWN - 1
            action_name = "DOWN"
        elif key == pygame.K_LEFT or key == pygame.K_a:
            action_type = self.env.LEFT - 1
            action_name = "LEFT"
        elif key == pygame.K_RIGHT or key == pygame.K_d:
            action_type = self.env.RIGHT - 1
            action_name = "RIGHT"
        elif key == pygame.K_b:
            action_type = self.env.BARRIER_MAKER - 1
            action_name = "BARRIER_MAKER"
            if self.selected_position is None:
                self.selected_position = (0, 0)
        elif key == pygame.K_h:
            action_type = self.env.HELLIFY - 1
            action_name = "HELLIFY"
            if self.selected_position is None:
                self.selected_position = (0, 0)
        
        if action_type is not None:
            # Execute action
            row, col = self.selected_position
            position_idx = row * self.env.n_cols + col
            action = (position_idx, action_type)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_action_info = info
            
            # Show feedback
            if info['valid_action']:
                msg = f"{action_name}: Reward={reward:.1f}"
                if info.get('chain_length', 0) > 1:
                    msg += f", Chain={info['chain_length']}"
                if info.get('lava_destroyed_this_step', 0) > 0:
                    msg += f", Destroyed={info['lava_destroyed_this_step']}"
                self._show_message(msg, self.colors['success'])
                
                self._add_animation_effect(info)
                
                # Update selection if box was destroyed
                if self.selected_position and self.env.grid[self.selected_position] != self.env.BOX:
                    self.selected_position = None
            else:
                self._show_message(f"{action_name}: Invalid action! Penalty applied", 
                                 self.colors['danger'])
        
        return terminated, truncated
    
    def _add_animation_effect(self, info: dict):
        """Add visual animation effects with opacity animations."""
        if info.get('lava_destroyed_this_step', 0) > 0:
            for r in range(self.env.n_rows):
                for c in range(self.env.n_cols):
                    if self.env.grid[r, c] == self.env.LAVA:
                        self.animation_cells.append((r, c, (255, 150, 0), 30))
                        # Add transformation opacity effect
                        self.transformation_effects[(r, c)] = {
                            'type': 'destruction',
                            'opacity': 255,
                            'duration': 30
                        }
        
        # Add opacity animation for moved boxes
        if info.get('moved_boxes', []):
            for pos in info.get('moved_boxes', []):
                if isinstance(pos, tuple) and len(pos) == 2:
                    self.moving_boxes[pos] = {
                        'opacity': 100,
                        'duration': 20,
                        'max_duration': 20
                    }
    
    def _update_animations(self):
        """Update animation timers with opacity effects."""
        self.animation_cells = [(r, c, color, timer - 1) 
                                for r, c, color, timer in self.animation_cells 
                                if timer > 1]
        
        # Update moving box opacity animations
        cells_to_remove = []
        for pos, anim_data in self.moving_boxes.items():
            # Fade opacity from 100 to 255 (brighten)
            progress = (anim_data['max_duration'] - anim_data['duration']) / anim_data['max_duration']
            anim_data['opacity'] = 100 + (155 * progress)  # Fade from 100 to 255
            anim_data['duration'] -= 1
            if anim_data['duration'] <= 0:
                cells_to_remove.append(pos)
        
        for pos in cells_to_remove:
            del self.moving_boxes[pos]
        
        # Update transformation effect opacity animations
        trans_to_remove = []
        for pos, trans_data in self.transformation_effects.items():
            # Fade opacity from 255 to 0 (fade out)
            progress = trans_data['duration'] / 30.0
            trans_data['opacity'] = int(255 * progress)
            trans_data['duration'] -= 1
            if trans_data['duration'] <= 0:
                trans_to_remove.append(pos)
        
        for pos in trans_to_remove:
            del self.transformation_effects[pos]
        
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = ""
    
    def _show_message(self, message: str, color: Tuple[int, int, int]):
        """Display a temporary message."""
        self.message = message
        self.message_color = color
        self.message_timer = 180
    
    def _render(self):
        """Render the game state."""
        self.window.fill(self.colors['background'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw sidebar
        self._draw_sidebar()
        
        # Draw HUD
        self._draw_hud()
        
        # Draw help overlay if enabled
        if self.show_help:
            self._draw_help_overlay()
    
    def _draw_grid(self):
        """Draw the game grid with textures."""
        # Grid background
        grid_rect = pygame.Rect(0, 0, self.grid_width, self.grid_height)
        pygame.draw.rect(self.window, self.colors['grid_bg'], grid_rect)
        
        # Highlight perfect squares
        perfect_square_cells = set()
        for size, top, left, _ in self.env.perfect_squares:
            for r in range(top, top + size):
                for c in range(left, left + size):
                    perfect_square_cells.add((r, c))
        
        # Draw cells
        for r in range(self.env.n_rows):
            for c in range(self.env.n_cols):
                cell_value = self.env.grid[r, c]
                
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Draw base texture
                if cell_value == self.env.LAVA:
                    self._draw_texture(self.window, 'lava', rect)
                elif cell_value == self.env.BOX:
                    self._draw_texture(self.window, 'floor', rect)
                    self._draw_texture(self.window, 'box', rect)
                elif cell_value == self.env.BARRIER:
                    self._draw_texture(self.window, 'barrier', rect)
                else:  # EMPTY
                    self._draw_texture(self.window, 'floor', rect)
                
                # Perfect square highlight
                if (r, c) in perfect_square_cells:
                    overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    overlay.fill((*self.colors['perfect_square'], 60))
                    self.window.blit(overlay, rect.topleft)
                
                # Animation overlay
                for anim_r, anim_c, anim_color, timer in self.animation_cells:
                    if anim_r == r and anim_c == c:
                        alpha = int(200 * (timer / 30.0))
                        overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        overlay.fill((*anim_color, alpha))
                        self.window.blit(overlay, rect.topleft)
                
                # Moving box opacity effect (fade-in glow)
                if (r, c) in self.moving_boxes:
                    box_anim = self.moving_boxes[(r, c)]
                    overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    # White overlay with dynamic opacity - subtle glow effect
                    glow_opacity = int(80 * (1 - box_anim['opacity'] / 255))
                    overlay.fill((255, 255, 200, glow_opacity))
                    self.window.blit(overlay, rect.topleft)
                
                # Transformation effect opacity (fade-out)
                if (r, c) in self.transformation_effects:
                    trans_effect = self.transformation_effects[(r, c)]
                    if trans_effect['type'] == 'destruction':
                        overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        # Orange-red destruction glow that fades out
                        overlay.fill((255, 100, 50, trans_effect['opacity'] // 2))
                        self.window.blit(overlay, rect.topleft)
                
                # Hover effect
                if self.hover_position and self.hover_position == (r, c):
                    self._draw_texture(self.window, 'hover', rect)
                
                # Selection
                if self.selected_position and self.selected_position == (r, c):
                    self._draw_texture(self.window, 'selected', rect)
                
                # Grid lines
                pygame.draw.rect(self.window, self.colors['border'], rect, 1)
                
                # Coordinates for boxes (smaller font if zoomed out)
                if cell_value == self.env.BOX and self.cell_size >= 40:
                    font = self.font_small if self.cell_size < 60 else self.font_medium
                    label = font.render(f"{r},{c}", True, (255, 255, 255))
                    label_rect = label.get_rect(center=rect.center)
                    
                    # Dark background for readability
                    bg_rect = label_rect.inflate(4, 2)
                    pygame.draw.rect(self.window, (0, 0, 0, 180), bg_rect)
                    self.window.blit(label, label_rect)
    
    def _draw_sidebar(self):
        """Draw the information sidebar."""
        sidebar_x = self.grid_width
        
        # Background
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.sidebar_width, self.grid_height)
        pygame.draw.rect(self.window, self.colors['sidebar'], sidebar_rect)
        pygame.draw.line(self.window, self.colors['border'], 
                        (sidebar_x, 0), (sidebar_x, self.grid_height), 2)
        
        y_offset = 15
        x_margin = 15
        
        # Title
        title = self.font_large.render("Shover-World", True, self.colors['accent'])
        self.window.blit(title, (sidebar_x + x_margin, y_offset))
        y_offset += 35
        
        # Separator
        pygame.draw.line(self.window, self.colors['border'],
                        (sidebar_x + x_margin, y_offset),
                        (sidebar_x + self.sidebar_width - x_margin, y_offset), 1)
        y_offset += 15
        
        # Game info
        info = self.env._get_info()
        
        info_lines = [
            ("Timestep", f"{info['timestep']}/{self.env.max_timestep}"),
            ("Stamina", f"{self.env.stamina:.0f}"),
            ("Boxes", f"{info['number_of_boxes']}"),
            ("Destroyed", f"{info['boxes_destroyed']}"),
            ("Squares", f"{len(self.env.perfect_squares)}"),
        ]
        
        for label, value in info_lines:
            # Label
            label_text = self.font_small.render(label + ":", True, self.colors['text_dim'])
            self.window.blit(label_text, (sidebar_x + x_margin, y_offset))
            
            # Value
            value_text = self.font_medium.render(value, True, self.colors['text'])
            self.window.blit(value_text, (sidebar_x + x_margin + 100, y_offset))
            y_offset += 25
        
        y_offset += 10
        
        # Stamina bar
        bar_width = self.sidebar_width - 2 * x_margin
        bar_height = 15
        bar_x = sidebar_x + x_margin
        
        # Background
        pygame.draw.rect(self.window, self.colors['panel_dark'],
                        (bar_x, y_offset, bar_width, bar_height))
        
        # Fill
        stamina_ratio = min(1.0, self.env.stamina / self.env.initial_stamina)
        fill_width = int(bar_width * stamina_ratio)
        
        if stamina_ratio > 0.6:
            bar_color = self.colors['success']
        elif stamina_ratio > 0.3:
            bar_color = self.colors['warning']
        else:
            bar_color = self.colors['danger']
        
        pygame.draw.rect(self.window, bar_color,
                        (bar_x, y_offset, fill_width, bar_height))
        
        # Border
        pygame.draw.rect(self.window, self.colors['border'],
                        (bar_x, y_offset, bar_width, bar_height), 1)
        
        y_offset += 30
        
        # Perfect squares
        if self.env.perfect_squares:
            pygame.draw.line(self.window, self.colors['border'],
                            (sidebar_x + x_margin, y_offset),
                            (sidebar_x + self.sidebar_width - x_margin, y_offset), 1)
            y_offset += 15
            
            ps_title = self.font_medium.render("Perfect Squares", True, self.colors['accent'])
            self.window.blit(ps_title, (sidebar_x + x_margin, y_offset))
            y_offset += 25
            
            for i, (size, top, left, creation_time) in enumerate(self.env.perfect_squares[:4]):
                age = self.env.timestep - creation_time
                ps_text = f"{size}x{size} ({top},{left}) age:{age}"
                text = self.font_small.render(ps_text, True, self.colors['text_dim'])
                self.window.blit(text, (sidebar_x + x_margin + 5, y_offset))
                y_offset += 20
        
        y_offset += 15
        
        # Controls section
        pygame.draw.line(self.window, self.colors['border'],
                        (sidebar_x + x_margin, y_offset),
                        (sidebar_x + self.sidebar_width - x_margin, y_offset), 1)
        y_offset += 15
        
        controls_title = self.font_medium.render("Controls", True, self.colors['accent'])
        self.window.blit(controls_title, (sidebar_x + x_margin, y_offset))
        y_offset += 25
        
        controls = [
            "Click - Select",
            "Arrows/WASD - Move",
            "H - Hellify",
            "B - Barrier",
            "R - Reset",
            "F1 - Help",
            "F11 - Fullscreen",
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, self.colors['text_dim'])
            self.window.blit(text, (sidebar_x + x_margin, y_offset))
            y_offset += 18
    
    def _draw_hud(self):
        """Draw the HUD at the bottom."""
        hud_y = self.grid_height
        
        # Background
        hud_rect = pygame.Rect(0, hud_y, self.window_width, self.hud_height)
        pygame.draw.rect(self.window, self.colors['hud'], hud_rect)
        pygame.draw.line(self.window, self.colors['border'],
                        (0, hud_y), (self.window_width, hud_y), 2)
        
        # Message
        if self.message:
            msg_text = self.font_medium.render(self.message, True, self.message_color)
            self.window.blit(msg_text, (15, hud_y + 15))
        
        # Last action info
        if self.last_action_info:
            y_pos = hud_y + 45
            
            details = [
                ("Valid", str(self.last_action_info.get('valid_action', False))),
                ("Chain", str(self.last_action_info.get('chain_length', 0))),
                ("Init Force", str(self.last_action_info.get('initial_force_charged', False))),
            ]
            
            x_pos = 15
            for label, value in details:
                text = self.font_small.render(f"{label}: {value}", True, self.colors['text_dim'])
                self.window.blit(text, (x_pos, y_pos))
                x_pos += 150
        
        # Cell size indicator
        zoom_text = self.font_small.render(f"Zoom: {self.cell_size}px (scroll to adjust)",
                                          True, self.colors['text_dim'])
        self.window.blit(zoom_text, (self.window_width - 250, hud_y + 15))
    
    def _draw_help_overlay(self):
        """Draw help overlay."""
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.window.blit(overlay, (0, 0))
        
        # Help panel
        panel_width = 500
        panel_height = 400
        panel_x = (self.window_width - panel_width) // 2
        panel_y = (self.window_height - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.window, self.colors['sidebar'], panel_rect)
        pygame.draw.rect(self.window, self.colors['accent'], panel_rect, 3)
        
        y = panel_y + 20
        x = panel_x + 20
        
        title = self.font_large.render("Help & Controls", True, self.colors['accent'])
        self.window.blit(title, (x, y))
        y += 40
        
        help_text = [
            "MOUSE CONTROLS:",
            "  • Click on box to select",
            "  • Scroll wheel to zoom in/out",
            "",
            "KEYBOARD CONTROLS:",
            "  • Arrow Keys / WASD - Move selected box",
            "  • H - Hellify (requires square > 2x2)",
            "  • B - Barrier Maker (requires square ≥ 2x2)",
            "  • R - Reset environment",
            "  • Space - Random action",
            "  • F1 - Toggle this help",
            "  • F11 - Toggle fullscreen",
            "  • ESC - Quit",
            "",
            "Press F1 to close"
        ]
        
        for line in help_text:
            text = self.font_small.render(line, True, self.colors['text'])
            self.window.blit(text, (x, y))
            y += 22


def main():
    """Main entry point."""
    print("=" * 70)
    print("Shover-World Enhanced GUI")
    print("=" * 70)
    print("\nTexture Requirements:")
    print("Place these 64x64px PNG images in 'assets/' folder:")
    print("  • lava.png - Red/orange lava texture")
    print("  • box.png - Wooden box texture")
    print("  • barrier.png - Stone/metal barrier texture")
    print("  • floor.png - Dark floor tile texture")
    print("  • selected.png - Selection overlay (transparent with green border)")
    print("  • hover.png - Hover overlay (transparent with yellow tint)")
    print("\nDefault textures will be used if images are not found.")
    print("=" * 70)
    
    # Create environment with smaller grid for better fitting
    env = ShoverWorldEnv(
        n_rows=10,
        n_cols=10,
        max_timestep=500,
        number_of_boxes=20,
        number_of_barriers=5,
        number_of_lavas=3,
        initial_stamina=2000.0,
        initial_force=40.0,
        unit_force=10.0,
        perf_sq_initial_age=5,
        seed=42,
        # map_path = 'D:\Shover\map_num.txt',
        # map_path = 'D:\Shover\map_sym.txt',
    )
    
    # Create and run GUI
    gui = EnhancedShoverWorldGUI(env, initial_cell_size=50)
    gui.run()


if __name__ == "__main__":
    main()
