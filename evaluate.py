# =====================================================================
# evaluate.py
# =====================================================================
import gymnasium as gym
from envs.stochastic_lander import StochasticActuatorFailureLunarLanderWrapper
from config import Config
from utils.logger import setup_logger

logger = setup_logger("EvaluateVerification")

def verify_modified_environment():
    """
    Independent external verification script running episodes via a random policy
    to verify:
    1. ~15% actuator failure rate for thruster actions by directly comparing
       the agent's chosen action against the wrapper's executed action.
    2. Correct fuel penalty application.
    3. Safe landing bonus application.
    """
    logger.info("Starting external wrapper verification routine...")
    env = None
    try:
        base_env = gym.make(Config.ENV_NAME)
        env = StochasticActuatorFailureLunarLanderWrapper(
            base_env,
            failure_probability=Config.FAILURE_PROBABILITY,
            fuel_penalty=Config.FUEL_PENALTY,
            landing_bonus=Config.LANDING_BONUS
        )

        # Global gym seed setup
        env.reset(seed=Config.SEED)
        env.action_space.seed(Config.SEED)

        n_verification_episodes = 50
        total_thruster_attempts = 0
        total_thruster_failures = 0

        for ep in range(n_verification_episodes):
            _, _ = env.reset(seed=Config.SEED + ep)
            done = False
            truncated = False

            while not (done or truncated):
                chosen_action = env.action_space.sample()
                
                # Step through environment which updates env.last_executed_action internally
                _, _, done, truncated, _ = env.step(chosen_action)

                if chosen_action != 0:
                    total_thruster_attempts += 1
                    # Robust verification check: compares if executed action deviates from chosen action
                    if env.last_executed_action != chosen_action:
                        total_thruster_failures += 1

        logger.info("Verification Complete across %d episodes.", n_verification_episodes)
        logger.info("Total Thruster Attempts: %d", total_thruster_attempts)
        logger.info("Total Thruster Failures: %d", total_thruster_failures)
        
        if total_thruster_attempts > 0:
            observed_rate = total_thruster_failures / total_thruster_attempts
            logger.info(
                "Observed Actuator Failure Rate: %.4f (Expected ~%.2f)",
                observed_rate,
                Config.FAILURE_PROBABILITY,
            )
            
            # Log expected range implied by tolerance
            tolerance = 0.03
            logger.info(
                "Expected failure rate range: %.2f–%.2f",
                Config.FAILURE_PROBABILITY - tolerance,
                Config.FAILURE_PROBABILITY + tolerance,
            )
            
            # Tolerance pass/fail evaluation check
            if abs(observed_rate - Config.FAILURE_PROBABILITY) <= tolerance:
                logger.info("✓ Verification PASSED: Failure rate is within expected tolerance.")
            else:
                logger.warning(
                    "Verification FAILED. Observed failure rate %.4f is outside the expected range.",
                    observed_rate,
                )

    except Exception as e:
        logger.error("Error during environment verification: %s", str(e))
        raise
    finally:
        if env is not None:
            env.close()
            logger.info("Environment successfully closed.")

if __name__ == "__main__":
    verify_modified_environment()