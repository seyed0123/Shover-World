Shover-World
=============

Quick start
-----------
1. Install dependencies (Python 3.8+ recommended):

2. Run the GUI:

   python3 main.py

AI solver
---------
- The project includes an A* solver implementation used by the GUI for automated play.
- To run the solver directly, import the solver class and create a `ShoverWorldEnv` instance.

Maps and assets
---------------
- Maps live in the `maps/` folder. Text map files can be loaded by path.
- Graphics are in `assets/`. Default placeholder textures are used when images are missing.

Project layout
--------------
- `environment.py` — main environment implementation
- `interactive_gui.py` — base GUI renderer
- `advanced_gui.py` — extended GUI with AI playback, recording, and stats
- `ai_solver.py` / `ai_solver_template.py` — A* solver implementations
- `main.py` — example runner
- `maps/`, `assets/` — data folders
