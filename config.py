# =====================================================================
# config.py
# =====================================================================
import torch

class Config:
    """
    Centralized configuration file for hyperparameters, environment settings,
    and paths to eliminate magic numbers across modules.
    """
    # Environment Settings
    ENV_NAME = "LunarLander-v3"
    FAILURE_PROBABILITY = 0.15
    FUEL_PENALTY = 0.3
    LANDING_BONUS = 50.0

    # Training Hyperparameters
    NUM_EPISODES = 800
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 50000
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQUENCY = 10
    MAX_GRAD_NORM = 1.0

    # Reproducibility & Hardware
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output Paths
    PLOT_DIR = "results/plots"
    PLOT_FILENAME = "assignment_performance_comparison.png"