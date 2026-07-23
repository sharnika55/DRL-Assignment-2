from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from envs.stochastic_lander import (
    THRUSTER_ACTIONS,
    StochasticActuatorFailureLunarLanderWrapper,
    is_valid_safe_landing,
)

# ==========================================
# CONFIGURATION & REPRODUCIBILITY SEEDING
# ==========================================
SEED = 42


def set_seed(seed):
  """Sets the random seed for random, numpy, torch, and action spaces."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class Config:
  ENV_NAME = "LunarLander-v3"
  FAILURE_RATE = 0.15
  FUEL_PENALTY = 0.3
  GAMMA = 0.99
  LR = 0.0005
  BATCH_SIZE = 64
  BUFFER_SIZE = 100000
  MIN_REPLAY_SIZE = 1000
  EPSILON_START = 1.0
  EPSILON_END = 0.01
  EPSILON_DECAY = 0.995
  NUM_EPISODES = 800  
  TARGET_UPDATE_FREQ = 10


# ==========================================
# NEURAL NETWORK & AGENT (DQN & DDQN)
# ==========================================
class QNetwork(nn.Module):
  """Neural network approximating action-value function Q(s, a)."""

  def __init__(self, state_dim, action_dim):
    """Initializes the Q-network layers."""
    super(QNetwork, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )

  def forward(self, x):
    """Forward pass through the network returning Q-values for all actions."""
    return self.fc(x)


class DQNAgent:
  """DQN or Double DQN Agent handling policy execution and experience replay updates."""

  def __init__(self, state_dim, action_dim, is_ddqn=False):
    """Initializes networks, optimizer, replay buffer, and configuration flags."""
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.is_ddqn = is_ddqn
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
    self.target_net = QNetwork(state_dim, action_dim).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(
        self.policy_net.parameters(), lr=Config.LR
    )
    self.memory = deque(maxlen=Config.BUFFER_SIZE)
    self.epsilon = Config.EPSILON_START

  def select_action(self, state, evaluate=False):
    """Selects an action using an epsilon-greedy policy."""
    if not evaluate and random.random() < self.epsilon:
      return random.randrange(self.action_dim)
    with torch.no_grad():
      state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
      q_values = self.policy_net(state_t)
      return q_values.argmax().item()

  def push(self, transition):
    """Pushes a new transition tuple into the replay buffer memory."""
    self.memory.append(transition)

  def sample(self, batch_size):
    """Samples a random batch of transitions from the replay buffer memory."""
    batch = random.sample(self.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.FloatTensor(np.array(states)).to(self.device),
        torch.LongTensor(actions).unsqueeze(1).to(self.device),
        torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
        torch.FloatTensor(np.array(next_states)).to(self.device),
        torch.FloatTensor(dones).unsqueeze(1).to(self.device),
    )

  def update(self):
    """Performs a gradient descent optimization step using Huber loss and gradient clipping."""
    if len(self.memory) < Config.MIN_REPLAY_SIZE:
      return

    states, actions, rewards, next_states, dones = self.sample(
        Config.BATCH_SIZE
    )

    current_q = self.policy_net(states).gather(1, actions)

    with torch.no_grad():
      if self.is_ddqn:
        best_actions = self.policy_net(next_states).argmax(
            dim=1, keepdim=True
        )
        max_next_q = self.target_net(next_states).gather(1, best_actions)
      else:
        max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]

      target_q = rewards + (Config.GAMMA * max_next_q * (1 - dones))

    # Using Huber Loss (SmoothL1Loss) for improved training stability
    loss = nn.SmoothL1Loss()(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
    self.optimizer.step()


# ==========================================
# EVALUATION & METRIC LOGGING UTILITIES
# ==========================================
def get_shared_validation_set(num_samples=50):
  """Collects a shared fixed validation set of states from the base environment for uniform Q-value tracking."""
  base_env = gym.make(Config.ENV_NAME)
  base_env.action_space.seed(SEED)
  val_states = []
  state, _ = base_env.reset(seed=SEED)
  for _ in range(num_samples):
    action = base_env.action_space.sample()
    next_state, _, terminated, truncated, _ = base_env.step(action)
    val_states.append(state)
    state = next_state
    if terminated or truncated:
      state, _ = base_env.reset()
  base_env.close()
  return torch.FloatTensor(np.array(val_states))


def evaluate_q_values(agent, val_states):
  """Computes average predicted Q-value across the shared validation set."""
  agent.policy_net.eval()
  with torch.no_grad():
    states_t = val_states.to(agent.device)
    q_values = agent.policy_net(states_t)
    avg_q = q_values.max(dim=1)[0].mean().item()
  agent.policy_net.train()
  return avg_q


def run_experiment(is_ddqn, use_wrapper, shared_val_states):
  """Executes full training for a specific configuration across the specified number of episodes."""
  exp_name = (
      f"{'DDQN' if is_ddqn else 'DQN'}_"
      f"{'Modified' if use_wrapper else 'Original'}"
  )
  print(f"\n--- Starting Experiment: {exp_name} ---")

  set_seed(SEED)

  base_env = gym.make(Config.ENV_NAME)
  base_env.action_space.seed(SEED)

  if use_wrapper:
    env = StochasticActuatorFailureLunarLanderWrapper(
        base_env,
        failure_rate=Config.FAILURE_RATE,
        fuel_penalty_weight=Config.FUEL_PENALTY,
        seed=SEED,
    )
  else:
    env = base_env

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  agent = DQNAgent(state_dim, action_dim, is_ddqn=is_ddqn)

  episode_rewards = []
  val_q_values = []
  success_moving_avg = []
  thruster_counts = []

  recent_successes = deque(maxlen=100)

  for episode in range(Config.NUM_EPISODES):
    state, _ = env.reset(seed=SEED + episode)
    terminated = False
    truncated = False
    total_reward = 0
    episode_thrusters = 0

    while not (terminated or truncated):
      action = agent.select_action(state)

      if action in THRUSTER_ACTIONS:
        episode_thrusters += 1

      next_state, reward, terminated, truncated, _ = env.step(action)
      agent.push((state, action, reward, next_state, float(terminated)))
      agent.update()

      state = next_state
      total_reward += reward

    if use_wrapper:
      is_success = is_valid_safe_landing(state, terminated, truncated)
    else:
      is_success = (
          terminated
          and abs(state[2]) < 0.1
          and abs(state[3]) < 0.1
          and abs(state[4]) < 0.1
          and state[6] == 1
          and state[7] == 1
      )

    recent_successes.append(1 if is_success else 0)

    agent.epsilon = max(Config.EPSILON_END, agent.epsilon * Config.EPSILON_DECAY)

    if (episode + 1) % Config.TARGET_UPDATE_FREQ == 0:
      agent.target_net.load_state_dict(agent.policy_net.state_dict())

    episode_rewards.append(total_reward)
    val_q_values.append(evaluate_q_values(agent, shared_val_states))
    success_moving_avg.append(
        np.mean(recent_successes) * 100.0 if recent_successes else 0.0
    )
    thruster_counts.append(episode_thrusters)

    if (episode + 1) % 50 == 0:
      print(
          f"Episode {episode+1}/{Config.NUM_EPISODES} | Reward:"
          f" {total_reward:.2f} | 100-Ep Success Rate:"
          f" {success_moving_avg[-1]:.1f}% | Thrusters:"
          f" {episode_thrusters} | Epsilon: {agent.epsilon:.2f}"
      )

  env.close()
  return {
      "rewards": episode_rewards,
      "q_values": val_q_values,
      "success_rate": success_moving_avg,
      "thrusters": thruster_counts,
  }


# ==========================================
# PLOTTING & MAIN EXECUTION
# ==========================================
def plot_results(results):
  """Generates and saves the 4 required comparison charts for part (d)."""
  epochs = range(1, Config.NUM_EPISODES + 1)

  plt.figure(figsize=(14, 10))

  plt.subplot(2, 2, 1)
  for name, data in results.items():
    plt.plot(epochs, data["rewards"], label=name, alpha=0.5)
  plt.title("Episode Reward vs Episode")
  plt.xlabel("Episode")
  plt.ylabel("Total Reward")
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 2)
  for name, data in results.items():
    plt.plot(epochs, data["q_values"], label=name)
  plt.title("Avg Predicted Q-Value (Shared Validation Set)")
  plt.xlabel("Episode")
  plt.ylabel("Q-Value")
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 3)
  for name, data in results.items():
    plt.plot(epochs, data["success_rate"], label=name)
  plt.title("100-Episode Moving Average Success Rate (%)")
  plt.xlabel("Episode")
  plt.ylabel("Success Rate (%)")
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 2, 4)
  for name, data in results.items():
    plt.plot(epochs, data["thrusters"], label=name, alpha=0.4)
  plt.title("Thruster Activations per Episode")
  plt.xlabel("Episode")
  plt.ylabel("Count")
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.savefig("experiment_comparison_plots.png")
  print("\nSaved all required comparison plots to 'experiment_comparison_plots.png'")
  plt.show()


def main():
  """Main execution entry point running all four required experiments and generating plots."""
  set_seed(SEED)
  shared_val_states = get_shared_validation_set()
  results = {}

  results["DQN - Original"] = run_experiment(
      is_ddqn=False, use_wrapper=False, shared_val_states=shared_val_states
  )
  results["DQN - Modified"] = run_experiment(
      is_ddqn=False, use_wrapper=True, shared_val_states=shared_val_states
  )
  results["DDQN - Original"] = run_experiment(
      is_ddqn=True, use_wrapper=False, shared_val_states=shared_val_states
  )
  results["DDQN - Modified"] = run_experiment(
      is_ddqn=True, use_wrapper=True, shared_val_states=shared_val_states
  )

  plot_results(results)


if __name__ == "__main__":
  main()