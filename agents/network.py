# =====================================================================
# agents/network.py
# =====================================================================
import torch
import torch.nn as nn
from utils.logger import setup_logger

logger = setup_logger("QNetwork")

class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron Q-Network architecture: 8 -> 128 -> 128 -> 4.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        try:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            logger.info("Initialized QNetwork with state_dim=%d, action_dim=%d, hidden_dim=%d", state_dim, action_dim, hidden_dim)
        except Exception as e:
            logger.error("Failed to initialize QNetwork: %s", str(e))
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)