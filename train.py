import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from envs.stochastic_lander import (
    StochasticActuatorFailureLunarLanderWrapper,
    is_valid_safe_landing,
)


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
  NUM_EPISODES = 500
  TARGET_UPDATE_FREQ = 10


class QNetwork(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(QNetwork, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )

  def forward(self, x):
    return self.fc(x)


class DQNAgent:

  def __init__(self, state_dim, action_dim):
    self.state_dim = state_dim
    self.action_dim = action_dim
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
    if not evaluate and random.random() < self.epsilon:
      return random.randrange(self.action_dim)
    with torch.no_grad():
      state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
      q_values = self.policy_net(state_t)
      return q_values.argmax().item()

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
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
    if len(self.memory) < Config.MIN_REPLAY_SIZE:
      return

    states, actions, rewards, next_states, dones = self.sample(
        Config.BATCH_SIZE
    )

    current_q = self.policy_net(states).gather(1, actions)
    with torch.no_grad():
      max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
      target_q = rewards + (Config.GAMMA * max_next_q * (1 - dones))

    loss = nn.MSELoss()(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


def evaluate_agent(agent, eval_env, num_episodes=50):
  """Evaluates the agent using the centralized safe-landing criteria."""
  successes = 0

  for _ in range(num_episodes):
    state, _ = eval_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
      action = agent.select_action(state, evaluate=True)
      next_state, _, terminated, truncated, _ = eval_env.step(action)
      state = next_state

      if is_valid_safe_landing(state, terminated, truncated):
        successes += 1
        break

  return (successes / num_episodes) * 100.0


def main():
  base_env = gym.make(Config.ENV_NAME)
  env = StochasticActuatorFailureLunarLanderWrapper(
      base_env,
      failure_rate=Config.FAILURE_RATE,
      fuel_penalty_weight=Config.FUEL_PENALTY,
  )

  eval_base = gym.make(Config.ENV_NAME)
  eval_env = StochasticActuatorFailureLunarLanderWrapper(
      eval_base,
      failure_rate=Config.FAILURE_RATE,
      fuel_penalty_weight=Config.FUEL_PENALTY,
  )

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n
  agent = DQNAgent(state_dim, action_dim)

  for episode in range(Config.NUM_EPISODES):
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
      action = agent.select_action(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      agent.push((state, action, reward, next_state, float(terminated)))
      agent.update()

      state = next_state
      total_reward += reward

    agent.epsilon = max(Config.EPSILON_END, agent.epsilon * Config.EPSILON_DECAY)

    if (episode + 1) % Config.TARGET_UPDATE_FREQ == 0:
      agent.target_net.load_state_dict(agent.policy_net.state_dict())

    if (episode + 1) % 10 == 0:
      success_rate = evaluate_agent(agent, eval_env)
      print(
          f"Episode {episode+1}/{Config.NUM_EPISODES} | Reward:"
          f" {total_reward:.2f} | Success Rate: {success_rate:.1f}% | Epsilon:"
          f" {agent.epsilon:.2f}"
      )

  env.close()
  eval_env.close()


if __name__ == "__main__":
  main()