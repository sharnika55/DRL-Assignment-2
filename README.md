# DRL-Assignment-2

## Deep Reinforcement Learning Assignment 2

This project implements and evaluates a modified **LunarLander-v3** task using **Gymnasium**.

The assignment includes:

- A custom environment wrapper with stochastic actuator failures
- Fuel usage penalties for thruster actions
- A strict safe-landing reward bonus
- Training and comparison of **DQN** vs **Double DQN**
- Automated verification of wrapper behavior
- Performance visualization across experiments

## Key Features

### Environment modifications

- **15% actuator failure probability** on thruster actions
- **Fuel penalty** of `0.3` for every thruster attempt
- **Safe landing bonus** of `+50` when strict landing criteria are satisfied

### Safe landing criteria

A landing is considered safe only if all of the following are true at terminal state:

- Both legs are in contact with the ground (`left_leg == 1` and `right_leg == 1`)
- `abs(horizontal_velocity) < 0.10`
- `abs(vertical_velocity) < 0.10`
- `abs(orientation_angle) < 0.10`

## Project Structure

```text
DRL-Assignment-2/
├── envs/
│   └── stochastic_lander.py      # Wrapper + safe-landing validation logic
├── results/
│   └── plots/                    # Optional folder for storing plots
├── train.py                      # Trains DQN/DDQN on original and modified envs
├── evaluate.py                   # Verifies wrapper behavior (part a)
└── README.md
```

## Requirements

- Python `3.10` or later
- `pip`
- Recommended: virtual environment

### Python dependencies

- `gymnasium[box2d]`
- `torch`
- `numpy`
- `matplotlib`

## Setup

### 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After activation, your prompt should start with `(.venv)`.

### 2. Install dependencies

```powershell
pip install gymnasium[box2d] torch numpy matplotlib
```

### 3. Verify installation

```powershell
python -c "import gymnasium, torch, numpy, matplotlib"
```

If no error is shown, setup is successful.

## How To Run

### Train all experiments

```powershell
python train.py
```

This runs four experiments:

- DQN - Original
- DQN - Modified
- DDQN - Original
- DDQN - Modified

During training, the script logs reward, success rate, thruster count, and epsilon progress.

### Verify wrapper behavior

```powershell
python evaluate.py
```

This script validates:

- Actuator failure rate is close to `0.15`
- Fuel-penalty math is correct
- Safe-landing bonus is applied only when criteria are met

## Expected Evaluation Output (Example)

```text
=== Starting Comprehensive Wrapper Verification ===
1. Actuator Failure Rate: 0.1498 (Expected ~0.15)
	-> PASSED
2. Fuel penalty (0.3) correct on NNNN/NNNN thruster steps; reward algebra correct on MMMM/MMMM steps
	-> PASSED
3. Safe landing reward = 59.70 (expected 59.70); unsafe landing reward = 10.00 (expected 10.00)
	-> PASSED
=== Verification Script Completed ===
```

Exact values may vary due to randomness.

## Output Artifacts

After training, one comparison figure is generated:

- `experiment_comparison_plots.png`

The figure includes four subplots:

- Episode reward vs episode
- Average predicted Q-value on a shared validation set
- 100-episode moving average success rate (%)
- Thruster activations per episode

## Notes

- The wrapper does **not** expose environment modifications through the returned `info` dictionary.
- Debug values used for verification are stored internally in the wrapper instance.