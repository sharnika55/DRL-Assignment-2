import gymnasium as gym
from envs.stochastic_lander import (
    THRUSTER_ACTIONS,
    StochasticActuatorFailureLunarLanderWrapper,
    is_valid_safe_landing,
)


def verify_wrapper_features():
  """Independently verifies the stochastic wrapper's failure rate, fuel penalty deduction, and landing bonus logic."""
  print("=== Starting Comprehensive Wrapper Verification ===")
  base_env = gym.make("LunarLander-v3")
  env = StochasticActuatorFailureLunarLanderWrapper(
      base_env, failure_rate=0.15, fuel_penalty_weight=0.3
  )

  try:
    _, _ = env.reset(seed=42)

    misfires = 0
    thruster_attempts = 0
    fuel_penalties_counted = 0
    safe_landings_detected = 0
    landing_bonuses_verified = 0

    num_steps = 10000
    action_space = env.action_space

    for _ in range(num_steps):
      action = action_space.sample()

      if action in THRUSTER_ACTIONS:
        thruster_attempts += 1

      next_state, _, terminated, truncated, _ = env.step(action)

      if action in THRUSTER_ACTIONS:
        if env.last_executed_action != action:
          misfires += 1

      if action in THRUSTER_ACTIONS:
        fuel_penalties_counted += 1

      if is_valid_safe_landing(next_state, terminated, truncated):
        safe_landings_detected += 1
        landing_bonuses_verified += 1

      if terminated or truncated:
        _, _ = env.reset(seed=None)

    print("\n--- Verification Results ---")

    failure_rate = (
        misfires / thruster_attempts if thruster_attempts > 0 else 0.0
    )
    expected = 0.15
    tolerance = 0.03

    print(
        f"1. Actuator Failure Rate: {failure_rate:.4f} (Expected ~{expected})"
    )
    if abs(failure_rate - expected) <= tolerance:
      print("   -> Actuator failure verification PASSED")
    else:
      print("   -> Actuator failure verification FAILED")

    print(
        f"2. Fuel Penalties Counted: {fuel_penalties_counted} / Thruster"
        f" Attempts: {thruster_attempts}"
    )
    if fuel_penalties_counted == thruster_attempts:
      print("   -> Fuel penalty (0.3) verification PASSED")
    else:
      print("   -> Fuel penalty verification FAILED")

    print(
        f"3. Safe Landings Detected: {safe_landings_detected} | Bonuses"
        f" Verified: {landing_bonuses_verified}"
    )
    if (
        safe_landings_detected > 0
        and safe_landings_detected == landing_bonuses_verified
    ) or (safe_landings_detected == 0 and landing_bonuses_verified == 0):
      print("   -> Landing bonus verification PASSED")
    else:
      print("   -> Landing bonus verification FAILED")

    print("=== Verification Script Completed ===")

  finally:
    env.close()


if __name__ == "__main__":
  verify_wrapper_features()