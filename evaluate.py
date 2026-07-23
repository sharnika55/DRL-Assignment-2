import gymnasium as gym
import numpy as np
from envs.stochastic_lander import (
    THRUSTER_ACTIONS,
    StochasticActuatorFailureLunarLanderWrapper,
    is_valid_safe_landing,
)


class _LandingStub(gym.Env):
  """Deterministic stub environment used to exercise the safe-landing bonus.

  Its step() always returns a terminal observation that either satisfies
  (safe=True) or violates (safe=False) every safe-landing criterion, so the
  +50 bonus logic can be demonstrated without relying on a rare random landing.
  """

  def __init__(self, safe=True):
    self.safe = safe
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
    )
    self.action_space = gym.spaces.Discrete(4)

  def reset(self, **kwargs):
    """Returns a zeroed initial observation."""
    return np.zeros(8, dtype=np.float32), {}

  def step(self, action):
    """Returns a terminal observation that is safe or unsafe per self.safe."""
    obs = np.zeros(8, dtype=np.float32)  # h_vel, v_vel, orientation = 0
    obs[6] = 1.0  # left leg contact
    obs[7] = 1.0  # right leg contact
    if not self.safe:
      obs[3] = 5.0  # large vertical velocity -> violates |v_vel| < 0.10
    return obs, 10.0, True, False, {}


def verify_failure_and_penalty(num_steps=10000):
  """Verifies the ~15% actuator failure rate and the 0.3 fuel-penalty deduction.

  Runs a random policy and, for every thruster attempt, checks the wrapper's
  reward algebra (reward == base - penalty + bonus) and that the applied fuel
  penalty is exactly 0.3 whether or not the engine misfired.
  """
  base_env = gym.make("LunarLander-v3")
  env = StochasticActuatorFailureLunarLanderWrapper(
      base_env, failure_rate=0.15, fuel_penalty_weight=0.3
  )

  misfires = 0
  thruster_attempts = 0
  penalty_ok = 0
  algebra_ok = 0
  total_checked = 0
  try:
    env.reset(seed=42)
    for _ in range(num_steps):
      action = env.action_space.sample()
      is_thruster = action in THRUSTER_ACTIONS
      if is_thruster:
        thruster_attempts += 1

      _, reward, terminated, truncated, _ = env.step(action)

      # Misfire detection via the internal executed-action attribute
      if is_thruster and env.last_executed_action != action:
        misfires += 1

      # Reward-algebra check (applies to every step)
      total_checked += 1
      expected = env.last_base_reward - env.last_fuel_penalty + env.last_bonus
      if abs(reward - expected) < 1e-6:
        algebra_ok += 1

      # Fuel-penalty check: 0.3 for thruster attempts, 0.0 otherwise
      expected_penalty = 0.3 if is_thruster else 0.0
      if is_thruster and abs(env.last_fuel_penalty - expected_penalty) < 1e-6:
        penalty_ok += 1

      if terminated or truncated:
        env.reset(seed=None)
  finally:
    env.close()

  rate = misfires / thruster_attempts if thruster_attempts > 0 else 0.0
  print(f"1. Actuator Failure Rate: {rate:.4f} (Expected ~0.15)")
  print("   ->", "PASSED" if abs(rate - 0.15) <= 0.03 else "FAILED")

  print(
      f"2. Fuel penalty (0.3) correct on {penalty_ok}/{thruster_attempts}"
      f" thruster steps; reward algebra correct on {algebra_ok}/{total_checked}"
      " steps"
  )
  penalty_pass = (
      thruster_attempts > 0
      and penalty_ok == thruster_attempts
      and algebra_ok == total_checked
  )
  print("   ->", "PASSED" if penalty_pass else "FAILED")


def verify_landing_bonus():
  """Demonstrates the +50 bonus is applied only on a valid safe landing.

  Uses a deterministic stub base env for both a safe and an unsafe terminal
  state and checks the wrapper's reward reflects the bonus exactly when the
  safe-landing criteria are met.
  """
  # Safe case: bonus MUST be applied (thruster action, no misfire).
  env_safe = StochasticActuatorFailureLunarLanderWrapper(
      _LandingStub(safe=True), failure_rate=0.0, fuel_penalty_weight=0.3
  )
  env_safe.reset(seed=0)
  obs_s, reward_s, term_s, trunc_s, _ = env_safe.step(2)
  expected_s = 10.0 - 0.3 + 50.0  # base - penalty + bonus
  safe_pass = (
      is_valid_safe_landing(obs_s, term_s, trunc_s)
      and env_safe.last_bonus == 50.0
      and abs(reward_s - expected_s) < 1e-6
  )
  env_safe.close()

  # Unsafe case: bonus MUST NOT be applied (do-nothing action, no penalty).
  env_unsafe = StochasticActuatorFailureLunarLanderWrapper(
      _LandingStub(safe=False), failure_rate=0.0, fuel_penalty_weight=0.3
  )
  env_unsafe.reset(seed=0)
  obs_u, reward_u, term_u, trunc_u, _ = env_unsafe.step(0)
  expected_u = 10.0 - 0.0 + 0.0  # base only
  unsafe_pass = (
      not is_valid_safe_landing(obs_u, term_u, trunc_u)
      and env_unsafe.last_bonus == 0.0
      and abs(reward_u - expected_u) < 1e-6
  )
  env_unsafe.close()

  print(
      f"3. Safe landing reward = {reward_s:.2f} (expected {expected_s:.2f});"
      f" unsafe landing reward = {reward_u:.2f} (expected {expected_u:.2f})"
  )
  print("   ->", "PASSED" if safe_pass and unsafe_pass else "FAILED")


def verify_wrapper_features():
  """Runs all three part-(a) verifications: failure rate, fuel penalty, bonus."""
  print("=== Starting Comprehensive Wrapper Verification ===")
  verify_failure_and_penalty()
  verify_landing_bonus()
  print("=== Verification Script Completed ===")


if __name__ == "__main__":
  verify_wrapper_features()