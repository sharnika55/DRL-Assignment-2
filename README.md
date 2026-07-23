# DRL-Assignment-2

## Deep Reinforcement Learning Assignment 2

This project implements a modified **LunarLander-v3** environment using **Gymnasium** with:

- Stochastic actuator failures (15%)
- Fuel consumption penalty
- Safe landing reward bonus
- DQN and Double DQN agents
- Performance comparison plots
- Wrapper verification script

---

# Project Structure

```
DRL-Assignment-2/
│
├── agents/
│   ├── agents.py
│   ├── network.py
│   └── replay_buffer.py
│
├── envs/
│   └── stochastic_lander.py
│
├── utils/
│   └── plotting.py
│
├── results/
│
├── config.py
├── train.py
├── evaluate.py
├── README.md
└── requirements.txt
```

---

# Prerequisites

- Python 3.10 or later
- pip
- Virtual Environment (recommended)

---

# Create Virtual Environment

Windows PowerShell

```powershell
python -m venv .venv
```

Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see

```
(.venv)
```

at the beginning of your terminal.

---

# Install Dependencies

```powershell
pip install -r requirements.txt
```

If requirements.txt is unavailable, install manually

```powershell
pip install gymnasium[box2d]
pip install torch
pip install numpy
pip install matplotlib
```

---

# Verify Installation

```powershell
python -c "import gymnasium, torch, numpy"
```

No errors indicate successful installation.

---

# Run Training

Train both DQN and Double DQN agents

```powershell
python train.py
```

Training will:

- Create the modified environment
- Train DQN
- Train Double DQN
- Save rewards
- Generate comparison plots

---

# Evaluate Trained Agents

```powershell
python evaluate.py
```

This verifies

- Actuator failure rate
- Fuel penalties
- Safe landing bonus

Expected output

```
=== Starting Comprehensive Wrapper Verification ===

--- Verification Results ---

1. Actuator Failure Rate: 0.1498
-> PASSED

2. Fuel Penalties Counted: xxxx / xxxx
-> PASSED

3. Safe Landings Detected: x
Independent Bonuses Verified: x
-> PASSED

=== Verification Script Completed ===
```

(The exact values will vary due to randomness.)

---

# Generated Results

After training, the following plot is generated.

```
assignment_performance_comparison.png
```

This figure compares

- DQN rewards
- Double DQN rewards
- Training performance
- Learning trends

---

# Features Implemented

### Environment Modifications

✔ 15% stochastic actuator failure

✔ Fuel penalty for every thruster action

✔ Safe landing bonus (+50 reward)

### Safe Landing Conditions

A landing is considered safe when

- Episode terminates successfully
- Not truncated
- Both landing legs touch the surface
- Horizontal velocity ≤ 0.10
- Vertical velocity ≤ 0.10
- Orientation angle ≤ 0.10 radians

---

# Wrapper Verification

The verification script checks

- Correct actuator failure probability
- Fuel penalty application
- Safe landing detection
- Landing reward bonus
- Requested vs executed actions

---

# Expected Output Files

```
assignment_performance_comparison.png
```

Additional model files may be generated depending on training configuration.

---

# Useful Commands

Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages

```powershell
pip install -r requirements.txt
```

Train agents

```powershell
python train.py
```

Evaluate wrapper

```powershell
python evaluate.py
```

Check installed packages

```powershell
pip list
```

Deactivate virtual environment

```powershell
deactivate
```

---

# Technologies Used

- Python
- Gymnasium
- NumPy
- PyTorch
- Matplotlib

---

# Author

K. Sharnika