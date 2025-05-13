# 2x2 Rubik’s Cube Solver with Q-Learning and A* Search

This is my **2x2 Rubik’s Cube Solver**, a project where I built an agent to tackle the tricky 2x2 Rubik’s Cube using Q-learning and A* search. The cube’s a small puzzle with a *relatively* massive state space, so I mixed reinforcement learning with a search algorithm to make it learn efficiently. I also added a curriculum to guide the learning process and some visualizations to show the results. If you’re into RL, puzzles, or just curious about cube-solving algorithms, this is for you.

## Table of Contents
- [Introduction](#introduction)
- [The Problem](#the-problem)
- [Cube Representation](#cube-representation)
- [Hybrid Agent](#hybrid-agent)
- [Curriculum Learning](#curriculum-learning)
- [Visualizations](#visualizations)
- [Training Results](#training-results)
- [Agent and Training Parameters](#agent-and-training-parameters)
- [Requirements](#requirements)
- [How to Use It](#how-to-use-it)
- [License](#license)

## Introduction
The 2x2 Rubik’s Cube, despite its size, has 3,674,160 possible configurations, making it a fun challenge for machine learning. I created an agent that learns to solve any scramble using Q-learning, boosted by A* search for smarter exploration. The project includes a custom cube environment, a hybrid learning agent, and visualizations to track progress and solutions. It’s a deep dive into combining RL with search to crack a classic puzzle.

## The Problem
Solving a 2x2 Rubik’s Cube means turning a scrambled state (mixed-up colors) into a solved state (each face one color) with minimal moves. The cube has 12 moves (clockwise and counterclockwise for e.g. U, F, L faces), and the hardest scrambles are 11 moves from solved, known as the “God’s number.” With millions of states, random moves won’t cut it, so I used Q-learning with A* search to learn an effective policy.

## Cube Representation
In my code (`Cube_Env.py`), the `RubiksCube2x2` class represents the cube in two formats:

- **Sticker State**: A 24-element NumPy array for all stickers (6 faces × 4 stickers). Colors are integers 0–5 (white, orange, green, red, blue, yellow). This is used for A* search and visualizations.
- **Compact State**: A tuple of:
  - `cubie_positions`: 8 numbers for each corner cubie’s position (0–7).
  - `cubie_orientations`: 8 numbers for each cubie’s orientation (0–2, as corners have three rotations).
  This compact state is used in the Q-table, focusing on cubie arrangements to reduce the state space.

The environment supports resetting, scrambling, applying moves, checking if solved, and getting states. The compact state keeps the Q-table manageable for learning.

## Hybrid Agent

My `RubikQLearningAgent` (in `Q_Agent.py`) combines Q-learning with A\* search to solve the cube efficiently:

- **Q-Learning**: The Q-table (a `defaultdict`) maps compact states to 12 move values. I update it with:

  $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

  where \$\alpha = 0.2\$ (learning rate), \$\gamma = 0.99\$ (discount factor). Rewards are +10 (plus a bonus for fewer steps) for solving, -0.1 per step, +0.05 per solved face, and -1 for timeout.

- **A* Search*\*: When a state isn’t in the Q-table (or during exploration with probability \$\epsilon\$), I use `batchedWeightedAStarSearch` (from `Search_Utils.py`) to find a path to solved. I only take the *first move* from A\*’s path, so the agent learns from its own experience rather than relying on search. The heuristic counts misplaced and misoriented cubies for efficiency.

- **Smaller State Space**: A* guidance means the Q-table doesn’t need all 3.7 million states. After 1M episodes, it holds 1,088,640 states—about 30% of the total.

The agent balances exploration (random or A*-guided moves) and exploitation (best Q-value), with \($\epsilon$\) decaying from 1.0 to 0.02.

## Curriculum Learning
To make learning smoother, I added a curriculum in the `train` method. The agent starts with 1-move scrambles and moves to 11-move scrambles when it hits a 70% solve rate for the current length. This was simple to implement—just a dictionary tracking solve rates—but it mirrors how humans learn: start easy, then tackle harder puzzles. The curriculum state is saved, so training can resume seamlessly.

## Visualizations
I included visualizations to show the agent’s progress and solutions, saved in the repo’s root directory:

- **Training Progress**: Plots rewards, steps, reward distribution, and state visits over episodes. Smoothed curves and log-scale histograms make trends clear, like how rewards climb as the agent learns.
  ![Training Progress](./training_progress.png)

- **Q-Value Heatmap**: Shows Q-values for the top 40 states, selected by their max Q-value. Rows are states, columns are moves, and colors highlight which moves the agent prefers.
  ![Q-Value Heatmap](./value_function_heatmap.png)

- **Solution Visualization**: Displays the cube’s states (solved → scrambled → each solution step) in a 2D cross layout. Colors match the cube’s faces, with labels (U, F, etc.) for clarity.
  ![Solution Visualization](./solution_steps_20250511_114903.png)

These are saved as `training_progress.png`, `value_function_heatmap.png`, and `solution_steps_<timestamp>.png` during training or solving.

## Training Results
I trained the agent for 1 million episodes, and it performs impressively. Evaluation results (100 episodes per scramble length, 1–11) from the logs:

- **Success Rates**:
  - Length 1: 100%
  - Length 2: 100%
  - Length 3: 100%
  - Length 4: 100%
  - Length 5: 99%
  - Length 6: 98%
  - Length 7: 97%
  - Length 8: 95%
  - Length 9: 92%
  - Length 10: 91%
  - Length 11: 91%
- **Win Rates**: Random agents solve ~0–2%, so these match the success rates.
- **Average Steps to Solve**: 6.12 steps across solved cases, showing efficiency.
- **Average Reward**: ~18.71 for the last 1000 episodes (length 11), driven by solve rewards.
- **Q-Table Size**: 1,088,640 states, ~30% of the 3.7M total, thanks to A*.
- **Training Time**: 3834 seconds (~64 minutes).

The agent masters short scrambles and handles deep 11-move scrambles well, a solid result for such a complex puzzle.

## Agent and Training Parameters
- **Agent**:
  - `epsilon`: 1.0, decays by 0.99995 per episode, minimum 0.02
  - `alpha`: 0.2
  - `gamma`: 0.99
  - `max_search_depth`: 15
- **Training**:
  - `num_episodes`: 1,000,000
  - `max_steps_per_episode`: 150
  - `scramble_lengths`: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  - `log_interval`: 1000
  - Reward: +10 (plus bonus) for solving, -0.1 per step, +0.05 per solved face, -1 for timeout

## Requirements
You need Python 3.8+ and these packages, listed in `requirements.txt`:

```
numpy>=1.21.0
h5py>=3.7.0
matplotlib>=3.5.0
tqdm>=4.62.0
```

Install with:
```bash
pip install -r requirements.txt
```

## How to Use It
The `main.py` script lets you evaluate or solve with the trained agent. Clone the repo, install dependencies, and try these:

1. **Evaluate and Solve**:
   ```bash
   python main.py
   ```
   Loads `rubik_final.h5` (or creates a new agent if missing), evaluates across lengths 1–11 (100 episodes each), solves a random 11-move scramble, and saves a visualization (`solution_steps_<timestamp>.png`). Results are in `rubik_agent.log`.

2. **Solve a Custom Scramble**:
   Edit `main.py` to set a scramble, e.g.:
   ```python
   scramble = ["U", "R", "F'"]
   solved, steps, solution = agent.solve(scramble, visualize=True)
   ```
   Run:
   ```bash
   python main.py
   ```
   It solves the scramble and saves a visualization.

3. **Train From Scratch**:
   The agent’s peaked at 1M episodes, but to retrain, uncomment in `main.py`:
   ```python
   agent.train(num_episodes=1000000, max_steps_per_episode=150, scramble_lengths=list(range(1, 12)), log_interval=1000)
   ```
   Run:
   ```bash
   python main.py
   ```
   Takes ~64 minutes for 1M episodes. Saves to `rubik_final.h5` with `training_progress.png` and `value_function_heatmap.png`.

4. **Resume Training**:
   If `rubik_final.h5` exists, training resumes (reset `last_episode` in `curriculum_state` if needed).

Check `rubik_agent.log` for details and visualizations for the agent’s performance.

## License
MIT License. Feel free to use or modify it—just don’t expect the cube to go easy on you!
