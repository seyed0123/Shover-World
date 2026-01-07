"""
advanced_gui_v2.py - Advanced GUI with dark mode, textures, and all extra features
Includes: Undo/Redo, Save/Load, Recording, Auto-play, Statistics
"""

import pygame
import numpy as np
import json
import pickle
import os
from datetime import datetime
from environment import ShoverWorldEnv
from gui import EnhancedShoverWorldGUI
from typing import Optional, Tuple, List, Dict
from AstarTemplate import AStarSolver
import threading
import queue
import time

SEARCH_TIME = None
STEMINA = None

class AdvancedShoverWorldGUI(EnhancedShoverWorldGUI):

    
    def __init__(self, env: ShoverWorldEnv, initial_cell_size: int = 50, fullscreen: bool = False):
        super().__init__(env, initial_cell_size, fullscreen)
        
        # AI fields 
        self.ai_plan = []
        self.ai_plan_idx = 0
        self.ai_play = False
        self.ai_busy = False
        self.ai_thread = None
        self.ai_message_queue = queue.Queue()


        self.history = []
        self.history_index = -1
        self.max_history = 100
        

        self.recording = False
        self.replay_data = []
        

        self.stats = {
            'total_actions': 0,
            'valid_actions': 0,
            'invalid_actions': 0,
            'boxes_destroyed_total': 0,
            'stamina_spent': 0,
            'max_chain': 0,
            'perfect_squares_made': 0,
            'session_start_time': datetime.now(),
        }


        self.auto_play = False
        self.auto_play_delay = 500
        self.last_auto_action_time = 0


        self.stats_panel_height = 120
        self.window_height += self.stats_panel_height
        self.window = pygame.display.set_mode((self.window_width, self.window_height), 
                                               pygame.RESIZABLE)
        pygame.display.set_caption("Shover-World - Advanced Edition")
        

        self.show_stats = True
        self.initialized = False

    def _build_state_snapshot(self):
        return {
            'grid': self.env.grid.copy(),
            'stamina': self.env.stamina,
            'timestep': self.env.timestep,
            'boxes_destroyed': self.env.boxes_destroyed,
            'box_stationary': dict(self.env.box_stationary),
            'perfect_squares': list(self.env.perfect_squares),
        }

    def _restore_state_snapshot(self, state: dict):
        self.env.grid = state['grid'].copy()
        self.env.stamina = state['stamina']
        self.env.timestep = state['timestep']
        self.env.boxes_destroyed = state['boxes_destroyed']
        self.env.box_stationary.clear()
        self.env.box_stationary.update(state['box_stationary'])
        self.env.perfect_squares = list(state['perfect_squares'])
    
    def _compute_ai_plan(self):
        if self.ai_busy:
            self._show_message("AI is already computing...", self.colors['warning'])
            return
        
        self.ai_busy = True
        self._show_message("AI planning (running in background)...", self.colors['accent'])
        
        # Run solver in separate thread
        self.ai_thread = threading.Thread(target=self._ai_worker_thread, daemon=True)
        self.ai_thread.start()

    def _ai_worker_thread(self):
        """Runs in background thread - does NOT touch GUI directly."""
        import logging
        

        logger = logging.getLogger('ai_solver')
        
        try:

            tmp = ShoverWorldEnv(
                n_rows=self.env.n_rows, n_cols=self.env.n_cols,
                max_timestep=self.env.max_timestep,
                number_of_boxes=self.env.number_of_boxes,
                number_of_barriers=self.env.number_of_barriers,
                number_of_lavas=self.env.number_of_lavas,
                initial_stamina=self.env.initial_stamina,
                initial_force=self.env.initial_force,
                unit_force=self.env.unit_force,
                perf_sq_initial_age=self.env.perf_sq_initial_age,
                seed=None
            )
            tmp.reset()
            

            tmp.grid = self.env.grid.copy()
            tmp.stamina = self.env.stamina
            tmp.timestep = self.env.timestep
            tmp.boxes_destroyed = self.env.boxes_destroyed
            tmp.box_stationary.clear()
            tmp.box_stationary.update(self.env.box_stationary)
            tmp.perfect_squares = list(self.env.perfect_squares)

            start = time.time()

            solver = AStarSolver(tmp, max_expansions=50_000, log_interval=500)
            plan = solver.solve()
            elapsed = time.time() - start
            logger.info(f"AI Solver: Path found with {len(plan)} steps in {elapsed:.3f} seconds." if plan else "AI Solver: No plan found.")
            global SEARCH_TIME
            SEARCH_TIME = elapsed

            self.ai_message_queue.put(('plan_ready', plan))
            
        except Exception as e:
            logger.error(f"AI solver crashed: {e}", exc_info=True)
            self.ai_message_queue.put(('plan_error', str(e)))

    def _check_ai_messages(self):
        """Called from main loop to process background thread results."""
        try:
            while True:
                msg_type, data = self.ai_message_queue.get_nowait()
                
                if msg_type == 'plan_ready':
                    self.ai_plan = data
                    self.ai_plan_idx = 0
                    self.ai_busy = False
                    
                    if self.ai_plan:
                        self._show_message(f"✓ AI plan ready: {len(self.ai_plan)} steps. Press O to play.", 
                                         self.colors['success'])
                    else:
                        self._show_message("✗ AI failed to find a plan.", self.colors['danger'])
                        
                elif msg_type == 'plan_error':
                    self.ai_busy = False
                    self._show_message(f"✗ AI error: {data}", self.colors['danger'])
                    
        except queue.Empty:
            pass


    def run(self):
        """Main game loop with advanced features."""
        obs, info = self.env.reset()
        self.initialized = True
        self._save_state()
        
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()

            self._check_ai_messages()


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._handle_mouse_click(event.pos)
                    elif event.button == 4:  # Scroll up
                        self._zoom(1)
                    elif event.button == 5:
                        self._zoom(-1)
                
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    terminated, truncated = self._handle_keyboard_advanced(event.key)
                    if terminated or truncated:
                        self._show_message("Episode ended! Press R to reset", 
                                         self.colors['danger'])
                        
            
            # Auto-play
            if self.auto_play and current_time - self.last_auto_action_time > self.auto_play_delay:
                if self.ai_play:
                    self._execute_ai_step()
                else:
                    self._execute_random_action()
                self.last_auto_action_time = current_time

            
            # Update animations
            self._update_animations()
            
            # Render
            self._render_advanced()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        self.env.close()
        pygame.quit()
    
    def _handle_keyboard_advanced(self, key: int) -> Tuple[bool, bool]:
        """Handle keyboard with advanced features."""
        mods = pygame.key.get_mods()
        ctrl_pressed = mods & pygame.KMOD_CTRL
        
        # --- ADVANCED CONTROLS ---
        
        # Undo/Redo
        if ctrl_pressed and key == pygame.K_z:
            self._undo()
            return False, False
        
        if ctrl_pressed and key == pygame.K_y:
            self._redo()
            return False, False
        
        # Save/Load
        if ctrl_pressed and key == pygame.K_s:
            self._save_game()
            return False, False
        
        if ctrl_pressed and key == pygame.K_l:
            self._load_game()
            return False, False
        
        # Recording
        if ctrl_pressed and key == pygame.K_r:
            self._toggle_recording()
            return False, False
        
        # Auto-play
        if key == pygame.K_p:
            self.auto_play = not self.auto_play
            status = "ON" if self.auto_play else "OFF"
            self._show_message(f"Auto-play: {status}", self.colors['success'])
            return False, False
        
        if key == pygame.K_n and not self.auto_play:
            self._execute_random_action()
            return False, False
        
        # Auto-play speed adjustment
        if key == pygame.K_EQUALS or key == pygame.K_PLUS:  # +
            self.auto_play_delay = max(100, self.auto_play_delay - 100)
            self._show_message(f"Auto-play speed: {1000//self.auto_play_delay} act/sec", 
                             self.colors['accent'])
            return False, False
        
        if key == pygame.K_MINUS:  # -
            self.auto_play_delay = min(2000, self.auto_play_delay + 100)
            self._show_message(f"Auto-play speed: {1000//self.auto_play_delay} act/sec", 
                             self.colors['accent'])
            return False, False
        
        # Toggle stats panel
        if key == pygame.K_TAB:
            self.show_stats = not self.show_stats
            return False, False
        
        # Handle reset specially
        if key == pygame.K_r and not ctrl_pressed:
            obs, info = self.env.reset()
            self.selected_position = None
            self._show_message("Environment reset!", self.colors['success'])
            self.history.clear()
            self.history_index = -1
            self._save_state()
            return False, False
            
        # --- STANDARD CONTROLS (Re-implemented to support recording) ---
        
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
            self._execute_random_action()
            return False, False
        
        # AI plan compute
        if key == pygame.K_g:
            self._compute_ai_plan()
            return False, False

        # Toggle AI playback (uses the existing timed auto-play loop)
        if key == pygame.K_o:
            if not self.ai_plan:
                self._show_message("No AI plan. Press G first.", self.colors['warning'])
            else:
                self.ai_play = not self.ai_play
                self.auto_play = self.ai_play  # reuse auto-play timer
                status = "ON" if self.ai_play else "OFF"
                self._show_message(f"AI playback: {status}", self.colors['success'])
            return False, False


        # Manual Moves
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
        
        # Execute Manual Move
        if action_type is not None:
            if self.selected_position is None:
                self._show_message("Please select a box first!", self.colors['warning'])
                return False, False
                
            row, col = self.selected_position
            position_idx = row * self.env.n_cols + col
            action = (position_idx, action_type)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.last_action_info = info
            
            # RECORDING LOGIC
            self._record_action(action, reward, info)
            self._update_stats(info, reward)
            
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
            
            # Save state after action
            if not (terminated or truncated) and self.initialized:
                self._save_state()
            
            return terminated, truncated

        return False, False
    
    def _record_action(self, action: Tuple, reward: float, info: dict):
        """Helper to record any action (manual or random)."""
        if self.recording:
            self.replay_data.append({
                'timestep': self.env.timestep,
                'action': [int(action[0]), int(action[1])],
                'reward': float(reward),
                'info': {k: v for k, v in info.items() 
                        if isinstance(v, (int, float, bool, str))}
            })

    def _execute_random_action(self):
        """Execute a random action."""
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.last_action_info = info
        self._update_stats(info, reward)
        
        # Record
        self._record_action(action, reward, info)
        
        self._add_animation_effect(info)
        self._save_state()

    def _save_state(self):
        """Save current state to history."""
        if not self.initialized:
            return
        
        state = {
            'grid': self.env.grid.copy(),
            'stamina': self.env.stamina,
            'timestep': self.env.timestep,
            'boxes_destroyed': self.env.boxes_destroyed,
            'box_stationary': dict(self.env.box_stationary),
            'perfect_squares': list(self.env.perfect_squares),
        }
        
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1
    
    def _undo(self):
        """Undo last action."""
        if self.history_index > 0:
            self.history_index -= 1
            self._restore_state(self.history[self.history_index])
            self._show_message("Undo", self.colors['accent'])
        else:
            self._show_message("Nothing to undo", self.colors['warning'])
    
    def _redo(self):
        """Redo next action."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._restore_state(self.history[self.history_index])
            self._show_message("Redo", self.colors['accent'])
        else:
            self._show_message("Nothing to redo", self.colors['warning'])
    
    def _restore_state(self, state: dict):
        """Restore environment to saved state."""
        self.env.grid = state['grid'].copy()
        self.env.stamina = state['stamina']
        self.env.timestep = state['timestep']
        self.env.boxes_destroyed = state['boxes_destroyed']
        self.env.box_stationary.clear()
        self.env.box_stationary.update(state['box_stationary'])
        self.env.perfect_squares = list(state['perfect_squares'])
    
    def _save_game(self):
        """Save game to file."""
        if not self.initialized or not self.history:
            self._show_message("No game state to save", self.colors['warning'])
            return
        
        os.makedirs("saves", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saves/shover_world_save_{timestamp}.pkl"
        
        save_data = {
            'state': self.history[self.history_index],
            'stats': self.stats,
            'env_params': {
                'n_rows': self.env.n_rows, 'n_cols': self.env.n_cols,
                'initial_stamina': self.env.initial_stamina,
                'initial_force': self.env.initial_force,
                'unit_force': self.env.unit_force,
                'max_timestep': self.env.max_timestep,
            }
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(save_data, f)
            self._show_message(f"Saved: {filename}", self.colors['success'])
        except Exception as e:
            self._show_message(f"Save failed: {str(e)}", self.colors['danger'])
    
    def _load_game(self):
        """Load game from file."""
        import glob
        if not os.path.exists("saves"):
            self._show_message("No saves folder found", self.colors['warning'])
            return
        saves = glob.glob("saves/shover_world_save_*.pkl")
        if not saves:
            self._show_message("No save files found", self.colors['warning'])
            return
        
        latest_save = max(saves)
        try:
            with open(latest_save, 'rb') as f: save_data = pickle.load(f)
            self.initialized = True
            self._restore_state(save_data['state'])
            self.stats = save_data.get('stats', self.stats)
            self._show_message(f"Loaded: {os.path.basename(latest_save)}", self.colors['success'])
        except Exception as e:
            self._show_message(f"Load failed: {str(e)}", self.colors['danger'])
    
    def _toggle_recording(self):
        """Toggle replay recording."""
        self.recording = not self.recording
        if self.recording:
            self.replay_data = []
            self._show_message("Recording started", self.colors['success'])
        else:
            os.makedirs("replays", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"replays/shover_world_replay_{timestamp}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump({
                        'replay_data': self.replay_data,
                        'stats': self.stats,
                        'env_params': {'n_rows': self.env.n_rows, 'n_cols': self.env.n_cols}
                    }, f, indent=2)
                self._show_message(f"Replay saved: {filename}", self.colors['success'])
            except Exception as e:
                self._show_message(f"Replay save failed: {str(e)}", self.colors['danger'])
    
    def _update_stats(self, info: dict, reward: float):
        """Update statistics."""
        self.stats['total_actions'] += 1
        if info.get('valid_action', False): self.stats['valid_actions'] += 1
        else: self.stats['invalid_actions'] += 1
        self.stats['boxes_destroyed_total'] = info['boxes_destroyed']
        if reward < 0: self.stats['stamina_spent'] += abs(reward)
        chain = info.get('chain_length', 0)
        if chain > self.stats['max_chain']: self.stats['max_chain'] = chain
        if len(self.env.perfect_squares) > 0:
            self.stats['perfect_squares_made'] = max(self.stats['perfect_squares_made'], len(self.env.perfect_squares))

    def _render_advanced(self):
        super()._render()
        if self.show_stats: 
            self._draw_stats_panel()
        self._draw_indicators()
        
        # AI computing indicator
        if self.ai_busy:
            self._draw_ai_computing_overlay()

    def _draw_ai_computing_overlay(self):
        """Show spinner while AI is computing."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.window.blit(overlay, (0, 0))
        
        # Spinner (fixed - no alpha in circle color)
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        
        angle = (pygame.time.get_ticks() / 5) % 360
        radius = 40
        for i in range(8):
            a = np.radians(angle + i * 45)
            x = center_x + int(radius * np.cos(a))
            y = center_y + int(radius * np.sin(a))
            # Vary brightness instead of alpha
            brightness = int(100 + 155 * (i / 8))
            color = (brightness, int(brightness * 0.7), 255)
            pygame.draw.circle(self.window, color, (x, y), 8)
        
        # Text
        text = self.font_large.render("AI Computing...", True, self.colors['accent'])
        text_rect = text.get_rect(center=(center_x, center_y + 70))
        self.window.blit(text, text_rect)
        
        tip = self.font_small.render("Check console for progress logs", True, self.colors['text_dim'])
        tip_rect = tip.get_rect(center=(center_x, center_y + 100))
        self.window.blit(tip, tip_rect)
        
        # DEBUG: Show queue status
        try:
            queue_size = self.ai_message_queue.qsize()
            debug_text = self.font_small.render(
                f"Queue: {queue_size} | Busy: {self.ai_busy} | Plan: {len(self.ai_plan)} steps", 
                True, self.colors['text_dim']
            )
            self.window.blit(debug_text, (center_x - 150, center_y + 130))
        except:
            pass

    def _execute_ai_step(self):
        """Execute one step from the AI plan."""
        if not self.ai_plan or self.ai_plan_idx >= len(self.ai_plan):
            self.ai_play = False
            self.auto_play = False
            self._show_message("AI playback finished.", self.colors['accent'])
            return

        action = self.ai_plan[self.ai_plan_idx]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action_info = info

        self._record_action(action, reward, info)
        self._update_stats(info, reward)
        self._add_animation_effect(info)
        self._save_state()

        self.ai_plan_idx += 1

        if terminated or truncated:
            self.ai_play = False
            self.auto_play = False
            self._show_message("Episode ended during AI playback.", self.colors['warning'])
            global STEMINA, SEARCH_TIME
            STEMINA = self.env.stamina - SEARCH_TIME//.2

            print(f"Remaining stemina: {STEMINA}, Search time: {SEARCH_TIME:.3f}")

    def _draw_stats_panel(self):
        panel_y = self.grid_height + self.hud_height
        pygame.draw.rect(self.window, self.colors['panel_dark'], (0, panel_y, self.window_width, self.stats_panel_height))
        pygame.draw.line(self.window, self.colors['border'], (0, panel_y), (self.window_width, panel_y), 2)

        y = panel_y + 10
        x_margin = 15
        title = self.font_medium.render("Statistics", True, self.colors['accent'])
        self.window.blit(title, (x_margin, y))
        y += 25

        stats_layout = [
            [("Actions", self.stats['total_actions']), ("Valid", self.stats['valid_actions']), ("Invalid", self.stats['invalid_actions'])],
            [("Destroyed", self.stats['boxes_destroyed_total']), ("Max Chain", self.stats['max_chain']), ("Squares Made", self.stats['perfect_squares_made'])],
            [("Stamina Spent", f"{self.stats['stamina_spent']:.0f}"), ("History", f"{self.history_index + 1}/{len(self.history)}"), ("Auto Speed", f"{1000//self.auto_play_delay}/s")]
        ]
        
        for col_idx, column in enumerate(stats_layout):
            x = x_margin + col_idx * 220
            y_local = y
            for label, value in column:
                self.window.blit(self.font_small.render(f"{label}:", True, self.colors['text_dim']), (x, y_local))
                self.window.blit(self.font_small.render(str(value), True, self.colors['text']), (x + 120, y_local))
                y_local += 20

        if 'session_start_time' in self.stats:
            elapsed = str(datetime.now() - self.stats['session_start_time']).split('.')[0]
            self.window.blit(self.font_small.render(f"Session: {elapsed}", True, self.colors['text_dim']), (self.window_width - 200, y))

    def _draw_indicators(self):
        indicator_y = 10
        indicator_x = self.grid_width - 180
        if self.recording:
            pulse = int(128 + 127 * abs(pygame.time.get_ticks() % 1000 - 500) / 500)
            pygame.draw.circle(self.window, (255, pulse // 2, pulse // 2), (indicator_x, indicator_y + 8), 8)
            self.window.blit(self.font_medium.render("REC", True, (255, 100, 100)), (indicator_x + 15, indicator_y))
            self.window.blit(self.font_small.render(f"[{len(self.replay_data)}]", True, self.colors['text_dim']), (indicator_x + 60, indicator_y + 2))
            indicator_y += 25

        if self.auto_play:
            play_color = self.colors['success']
            pygame.draw.polygon(self.window, play_color, [(indicator_x-5, indicator_y), (indicator_x-5, indicator_y+15), (indicator_x+10, indicator_y+7)])
            self.window.blit(self.font_medium.render("AUTO-PLAY", True, play_color), (indicator_x + 15, indicator_y - 2))
            self.window.blit(self.font_small.render(f"{1000//self.auto_play_delay} act/s", True, self.colors['text_dim']), (indicator_x + 110, indicator_y + 2))

    def _draw_help_overlay(self):
        """Draw enhanced help overlay."""
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        self.window.blit(overlay, (0, 0))
        
        # Help panel
        panel_width = 600
        panel_height = 600
        panel_x = (self.window_width - panel_width) // 2
        panel_y = (self.window_height - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.window, self.colors['sidebar'], panel_rect)
        pygame.draw.rect(self.window, self.colors['accent'], panel_rect, 3)
        
        y = panel_y + 15
        x = panel_x + 20
        
        title = self.font_large.render("Advanced Controls & Help", True, self.colors['accent'])
        self.window.blit(title, (x, y))
        y += 35
        
        help_sections = [
            ("BASIC CONTROLS:", [
                "Click - Select box",
                "Arrow/WASD - Move box",
                "H - Hellify | B - Barrier Maker",
                "R - Reset | Space - Random",
            ]),
            ("ADVANCED FEATURES:", [
                "Ctrl+Z - Undo | Ctrl+Y - Redo",
                "Ctrl+S - Save | Ctrl+L - Load",
                "Ctrl+R - Toggle recording",
                "P - Auto-play | N - Next step",
                "+/- - Adjust auto-play speed",
            ]),
            ("VIEW CONTROLS:", [
                "Scroll - Zoom in/out",
                "F1 - Toggle help",
                "F11 - Fullscreen",
                "Tab - Toggle stats panel",
                "ESC - Quit",
            ]),
            ("AI CONTROLS:", [
                "G - Generate AI plan (background)",
                "O - Toggle AI playback",
                "+/- - Adjust playback speed",
            ]),

        ]
        
        for section_title, items in help_sections:
            section_text = self.font_medium.render(section_title, True, self.colors['accent'])
            self.window.blit(section_text, (x, y))
            y += 25
            
            for item in items:
                item_text = self.font_small.render(f"  • {item}", True, self.colors['text'])
                self.window.blit(item_text, (x + 10, y))
                y += 20
            
            y += 10
        
        # Close instruction
        close_text = self.font_small.render("Press F1 to close", 
                                           True, self.colors['text_dim'])
        self.window.blit(close_text, (panel_x + panel_width // 2 - 60, 
                                      panel_y + panel_height - 30))


def main():
    """Main entry point for advanced GUI."""
    print("=" * 70)
    print("Shover-World Advanced Edition (FIXED RECORDING)")
    print("=" * 70)
    
    # Use correct map path or comment out to use random generation
    env = ShoverWorldEnv(
        n_rows=11,
        n_cols=15,
        max_timestep=1000,
        number_of_boxes=20,
        number_of_barriers=8,
        number_of_lavas=3,
        initial_stamina=1000.0,
        initial_force=40.0,
        unit_force=10.0,
        perf_sq_initial_age=5,
        # map_path="maps/t.txt",  # Uncomment when file exists in correct location

    )
    
    gui = AdvancedShoverWorldGUI(env, initial_cell_size=45)
    gui.run()


if __name__ == "__main__":
    main()
