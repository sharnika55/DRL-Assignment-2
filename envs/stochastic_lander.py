# =====================================================================
# envs/stochastic_lander.py
# =====================================================================
import random
import gymnasium as gym
from utils.logger import setup_logger

logger = setup_logger("StochasticLanderEnv")

class StochasticActuatorFailureLunarLanderWrapper(gym.Wrapper):
    """
    Custom wrapper implementing gym.Wrapper for LunarLander-v3.
    Modifies action execution (15% failure for actions 1, 2, 3), applies 
    fuel penalties based on the agent's chosen action, and assigns conditional 
    landing bonuses strictly adhering to the assignment requirements.
    """
    def __init__(self, env: gym.Env, failure_probability: float = 0.15, fuel_penalty: float = 0.3, landing_bonus: float = 50.0):
        super(StochasticActuatorFailureLunarLanderWrapper, self).__init__(env)
        self.failure_probability = failure_probability
        self.fuel_penalty = fuel_penalty
        self.landing_bonus_val = landing_bonus
        
        # Internal tracking attribute used solely by external verification scripts
        self.last_executed_action = None
        logger.info("Initialized StochasticActuatorFailureLunarLanderWrapper with failure_prob=%.2f", failure_probability)

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except Exception as e:
            logger.error("Error occurred during environment reset: %s", str(e))
            raise

    def step(self, action):
        """
        Executes a step with stochastic failure injection, action validation, and modified reward computation.
        """
        try:
            # Action validation check
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid action {action} passed to environment step function.")

            # Step 1: Store the agent's originally selected action (a)
            original_action = int(action)

            # Step 2: Simulate Intermittent Engine Failure
            if original_action == 0:
                executed_action = 0
            else:
                rand_num = random.random()
                if rand_num < self.failure_probability:
                    executed_action = 0  # Replaced by Do Nothing
                else:
                    executed_action = original_action

            # Store executed action for wrapper verification access
            self.last_executed_action = executed_action

            # Step 3: Execute the action in the base environment
            observation, base_reward, terminated, truncated, info = self.env.step(executed_action)

            # Step 4: Compute the Modified Reward
            # R = R_base - 0.3 * 1_{a != 0} + B
            indicator_nonzero = 1 if original_action != 0 else 0
            
            # Step 5: Check Safe Landing Bonus (B) exactly per assignment specs:
            # - terminated == True and truncated == False
            # - observation[6] == 1 (left leg contact)
            # - observation[7] == 1 (right leg contact)
            # - vertical speed (observation[3]) > -0.2
            # - orientation angle (abs(observation[4])) < 0.1 radians
            bonus = 0.0
            if terminated and not truncated:
                v_vel = observation[3]
                orientation = observation[4]
                left_leg = observation[6]
                right_leg = observation[7]

                if (left_leg == 1.0 and right_leg == 1.0 and
                    v_vel > -0.2 and abs(orientation) < 0.10):
                    bonus = self.landing_bonus_val

            modified_reward = base_reward - (self.fuel_penalty * indicator_nonzero) + bonus

            # Step 6: Return output (no extra info dictionary keys added per specifications)
            return observation, modified_reward, terminated, truncated, info

        except Exception as e:
            logger.error("Exception encountered during wrapper step execution: %s", str(e))
            raise