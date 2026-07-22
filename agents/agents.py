# =====================================================================
# agents/agents.py
# =====================================================================
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.network import QNetwork
from agents.replay_buffer import ReplayBuffer
from config import Config
from utils.logger import setup_logger

logger = setup_logger("DQNAgent")

class DQNAgent:
    """
    Unified agent supporting both DQN and DDQN variants. 
    The target Q-value computation distinguishes DQN from DDQN.
    """
    def __init__(self, state_dim: int, action_dim: int, is_ddqn: bool = False):
        try:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.is_ddqn = is_ddqn

            # Set random seeds for reproducibility
            random.seed(Config.SEED)
            np.random.seed(Config.SEED)
            torch.manual_seed(Config.SEED)

            self.device = Config.DEVICE
            logger.info("Using device: %s for agent (DDQN=%s)", self.device, is_ddqn)

            # Networks & Optimizer
            self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
            self.memory = ReplayBuffer(Config.BUFFER_CAPACITY)
            logger.info("Successfully initialized Agent")
        except Exception as e:
            logger.error("Error initializing Agent: %s", str(e))
            raise

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action using an epsilon-greedy exploration strategy.
        """
        try:
            if random.random() < epsilon:
                return random.randrange(self.action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = self.policy_net(state_t)
                    return int(q_values.argmax().item())
        except Exception as e:
            logger.error("Error selecting action: %s", str(e))
            raise

    def update_model(self):
        """
        Performs a gradient update step using Huber loss and gradient clipping.
        """
        try:
            if len(self.memory) < Config.BATCH_SIZE:
                return

            states, actions, rewards, next_states, dones = self.memory.sample(Config.BATCH_SIZE)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            current_q_values = self.policy_net(states_t).gather(1, actions_t)

            with torch.no_grad():
                if self.is_ddqn:
                    # Double DQN Target Calculation
                    best_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                    next_q_values = self.target_net(next_states_t).gather(1, best_actions)
                else:
                    # Standard DQN Target Calculation
                    next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

                target_q_values = rewards_t + (Config.GAMMA * next_q_values * (1.0 - dones_t))

            loss_fn = nn.SmoothL1Loss()  # Huber Loss
            loss = loss_fn(current_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=Config.MAX_GRAD_NORM)
            self.optimizer.step()

        except Exception as e:
            logger.error("Error updating model weights: %s", str(e))
            raise

    def update_target_network(self):
        try:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        except Exception as e:
            logger.error("Error updating target network: %s", str(e))
            raise