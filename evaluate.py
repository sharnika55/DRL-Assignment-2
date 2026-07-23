import gymnasium as gym
from envs.stochastic_lander import (
    THRUSTER_ACTIONS,
    StochasticActuatorFailureLunarLanderWrapper,
)


def verify_wrapper_features():
  print("=== Starting Comprehensive Wrapper Verification ===")
  base_env = gym.make("LunarLander-v3")
  env = StochasticActuatorFailureLunarLanderWrapper(base_env, failure_rate=0.15)

  try:
    # Capture reset return values to avoid linter warnings
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

      # Discard unused state and reward variables cleanly
      _, _, terminated, truncated, info = env.step(action)

      # Check actual misfires using info dictionary and requested action
      if action in THRUSTER_ACTIONS:
        if info.get("executed_action") != info.get("requested_action"):
          misfires += 1

      # Count fuel penalties using the wrapper's fuel_penalty_applied flag
      if info.get("fuel_penalty_applied", False):
        fuel_penalties_counted += 1

      # Track safe landings and explicit landing bonuses independently
      if info.get("is_safe_landing", False):
        safe_landings_detected += 1

      if info.get("landing_bonus_applied", False):
        landing_bonuses_verified += 1

      if terminated or truncated:
        _, _ = env.reset(seed=None)

    print("\n--- Verification Results ---")

    # 1. Verify Failure Rate (Expected ~15% with strict ±3% tolerance)
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

    # 2. Verify Fuel Penalty
    print(
        f"2. Fuel Penalties Counted: {fuel_penalties_counted} / Thruster"
        f" Attempts: {thruster_attempts}"
    )
    if fuel_penalties_counted == thruster_attempts:
      print("   -> Fuel penalty verification PASSED")
    else:
      print("   -> Fuel penalty verification FAILED")

    # 3. Verify Landing Bonus Independently
    print(
        f"3. Safe Landings Detected: {safe_landings_detected} | Independent"
        f" Bonuses Verified: {landing_bonuses_verified}"
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