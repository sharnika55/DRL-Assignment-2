import gymnasium as gym
import numpy as np

THRUSTER_ACTIONS = {1, 2, 3}


class StochasticActuatorFailureLunarLanderWrapper(gym.Wrapper):
  """Custom wrapper for LunarLander introducing stochastic actuator failures,

  fuel penalties, and strict multi-variable safe-landing criteria.
  """

  def __init__(
      self, env, failure_rate=0.15, fuel_penalty_weight=0.01, seed=None
  ):
    super().__init__(env)
    self.failure_rate = failure_rate
    self.fuel_penalty_weight = fuel_penalty_weight
    self.np_random, _ = gym.utils.seeding.np_random(seed)
    self.last_executed_action = None

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    # 1. Apply 15% stochastic failure rate for thruster actions
    actual_action = action
    if action in THRUSTER_ACTIONS:
      if self.np_random.random() < self.failure_rate:
        actual_action = 0  # Misfire / do nothing

    self.last_executed_action = actual_action

    # 2. Step the base environment with the actual action
    next_state, reward, terminated, truncated, info = self.env.step(
        actual_action
    )

    # 3. Fuel penalty applied on every thruster attempt, including misfires
    fuel_penalty = 0.0
    if action in THRUSTER_ACTIONS:
      fuel_penalty = self.fuel_penalty_weight
      reward -= fuel_penalty

    # 4. Strict Safe-Landing Criteria Check
    h_vel = next_state[2]
    v_vel = next_state[3]
    orientation = next_state[4]
    left_leg = next_state[6]
    right_leg = next_state[7]

    is_safe_landing = (
        terminated
        and not truncated
        and left_leg == 1
        and right_leg == 1
        and abs(h_vel) <= 0.10
        and abs(v_vel) <= 0.10
        and abs(orientation) <= 0.10
    )

    # 5. Apply +50 reward bonus specifically for a valid safe landing
    landing_bonus_applied = False
    if is_safe_landing:
      reward += 50.0
      landing_bonus_applied = True

    # Populate info dictionary with explicit verification flags and requested action
    info["is_safe_landing"] = is_safe_landing
    info["fuel_penalty_applied"] = fuel_penalty > 0.0
    info["landing_bonus_applied"] = landing_bonus_applied
    info["executed_action"] = actual_action
    info["requested_action"] = action

    return next_state, reward, terminated, truncated, info