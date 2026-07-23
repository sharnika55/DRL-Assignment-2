import gymnasium as gym
import numpy as np

THRUSTER_ACTIONS = (1, 2, 3)


def is_valid_safe_landing(state, terminated, truncated):
  """Centralized safe landing check using strict < 0.10 thresholds."""
  if not terminated or truncated:
    return False

  h_vel = state[2]
  v_vel = state[3]
  orientation = state[4]
  left_leg = state[6]
  right_leg = state[7]

  return (
      left_leg == 1
      and right_leg == 1
      and abs(h_vel) < 0.10
      and abs(v_vel) < 0.10
      and abs(orientation) < 0.10
  )


class StochasticActuatorFailureLunarLanderWrapper(gym.Wrapper):
  """Custom wrapper for LunarLander introducing stochastic actuator failures,

  a 0.3 fuel penalty, and centralized safe-landing criteria.
  """

  def __init__(
      self, env, failure_rate=0.15, fuel_penalty_weight=0.3, seed=None
  ):
    super().__init__(env)
    self.failure_rate = failure_rate
    self.fuel_penalty_weight = fuel_penalty_weight
    self.np_random, _ = gym.utils.seeding.np_random(seed)
    self.last_executed_action = None
    # Debug-only attributes exposed for external verification. These are stored
    # on the wrapper instance and are deliberately NEVER added to the returned
    # info dict, so the agent still receives no indication of any modification.
    self.last_base_reward = None
    self.last_fuel_penalty = None
    self.last_bonus = None

  def reset(self, **kwargs):
    """Resets the underlying environment, forwarding all keyword arguments."""
    return self.env.reset(**kwargs)

  def step(self, action):
    """Executes one environment step with stochastic actuator failure.

    Injects a 15% chance of replacing a thruster action with Do-Nothing,
    subtracts the 0.3 fuel penalty on every thruster attempt (including
    misfires), and adds a +50 bonus on a verified safe landing. The base
    reward, applied penalty, and applied bonus are recorded as debug
    attributes for verification but are not written to the info dict.
    """
    # 1. Apply 15% stochastic failure rate for thruster actions
    actual_action = action
    if action in THRUSTER_ACTIONS:
      if self.np_random.random() < self.failure_rate:
        actual_action = 0  # Misfire / do nothing

    self.last_executed_action = actual_action

    # 2. Step the base environment with the actual action
    next_state, base_reward, terminated, truncated, info = self.env.step(
        actual_action
    )

    # 3. Fuel penalty (0.3) on every thruster attempt, regardless of misfire
    fuel_penalty = (
        self.fuel_penalty_weight if action in THRUSTER_ACTIONS else 0.0
    )

    # 4. Strict Safe-Landing Criteria Check using shared function
    bonus = (
        50.0 if is_valid_safe_landing(next_state, terminated, truncated) else 0.0
    )

    # 5. Compose the modified reward: R = R_base - 0.3 * 1_{a != 0} + B
    reward = base_reward - fuel_penalty + bonus

    # Record debug values for external verification (NOT added to info dict)
    self.last_base_reward = base_reward
    self.last_fuel_penalty = fuel_penalty
    self.last_bonus = bonus

    # Return info completely unmodified as per assignment specification
    return next_state, reward, terminated, truncated, info