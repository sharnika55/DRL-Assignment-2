# =====================================================================
# train.py
# =====================================================================
import numpy as np
import torch
import gymnasium as gym
from config import Config
from envs.stochastic_lander import StochasticActuatorFailureLunarLanderWrapper
from agents.agents import DQNAgent
from utils.plotting import plot_performance_results
from utils.logger import setup_logger

logger = setup_logger("TrainPipeline")

def run_training_experiment(is_modified: bool, is_ddqn: bool) -> dict:
    """
    Executes training for a given environment configuration and agent type,
    returning performance metrics for comparative analysis.
    """
    env_type = "Modified" if is_modified else "Original"
    agent_type = "DDQN" if is_ddqn else "DQN"
    logger.info("Starting training experiment: Env=%s, Agent=%s, Episodes=%d",
                env_type, agent_type, Config.NUM_EPISODES)
    try:
        base_env = gym.make(Config.ENV_NAME)
        env = StochasticActuatorFailureLunarLanderWrapper(base_env) if is_modified else base_env

        env.reset(seed=Config.SEED)
        env.action_space.seed(Config.SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, is_ddqn=is_ddqn)

        # Fixed validation states for consistent Q-value tracking
        validation_states = []
        val_obs, _ = env.reset(seed=Config.SEED)
        validation_states.append(val_obs)
        for _ in range(9):
            val_obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                val_obs, _ = env.reset()
            validation_states.append(val_obs)
        validation_states = np.array(validation_states, dtype=np.float32)

        epsilon = Config.EPSILON_START
        episode_rewards = []
        avg_q_values = []
        landing_successes = []
        moving_avg_success_rate = []
        thruster_counts = []

        for episode in range(1, Config.NUM_EPISODES + 1):
            _, _ = env.reset(seed=Config.SEED + episode)
            episode_reward = 0.0
            done = False
            truncated = False
            thruster_activations = 0

            # Retrieve the starting state from the reset result tuple
            state, _ = env.reset(seed=Config.SEED + episode)

            while not (done or truncated):
                action = agent.select_action(state, epsilon)
                if action != 0:
                    thruster_activations += 1

                next_state, reward, done, truncated, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, float(done or truncated))
                agent.update_model()

                state = next_state
                episode_reward += reward

            epsilon = max(Config.EPSILON_MIN, epsilon * Config.EPSILON_DECAY)

            if episode % Config.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            # Landing success tracking per assignment specifications
            success = 0.0
            if done and not truncated:
                v_vel = state[3]
                orientation = state[4]
                left_leg = state[6]
                right_leg = state[7]
                if (left_leg == 1.0 and right_leg == 1.0 and
                    v_vel > -0.2 and abs(orientation) < 0.10):
                    success = 1.0

            landing_successes.append(success)
            window_slice = landing_successes[-100:]
            current_success_rate = sum(window_slice) / len(window_slice)
            moving_avg_success_rate.append(current_success_rate)

            # Validation Q-value computation using torch.as_tensor
            with torch.no_grad():
                val_states_t = torch.as_tensor(validation_states, dtype=torch.float32, device=agent.device)
                q_preds = agent.policy_net(val_states_t)
                mean_q = q_preds.max(dim=1)[0].mean().item()

            episode_rewards.append(episode_reward)
            avg_q_values.append(mean_q)
            thruster_counts.append(thruster_activations)

            if episode % 100 == 0:
                logger.info("Episode %d/%d | Reward: %.2f | Epsilon: %.2f | Mean Q: %.2f | Success Rate: %.2f",
                            episode, Config.NUM_EPISODES, episode_reward, epsilon, mean_q, current_success_rate)

        env.close()
        return {
            "rewards": episode_rewards,
            "q_values": avg_q_values,
            "success_rate": moving_avg_success_rate,
            "thruster_counts": thruster_counts
        }

    except Exception as e:
        logger.error("Error during training execution: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        logger.info("=== Starting Full Modular Training Pipeline ===")
        results = {}

        # 1. DQN - Original Environment
        results["DQN_Original"] = run_training_experiment(is_modified=False, is_ddqn=False)

        # 2. DDQN - Original Environment
        results["DDQN_Original"] = run_training_experiment(is_modified=False, is_ddqn=True)

        # 3. DQN - Modified Environment
        results["DQN_Modified"] = run_training_experiment(is_modified=True, is_ddqn=False)

        # 4. DDQN - Modified Environment
        results["DDQN_Modified"] = run_training_experiment(is_modified=True, is_ddqn=True)

        # Generate comparison evaluation plots
        plot_performance_results(results)

    except Exception as e:
        logger.critical("Critical error in training pipeline: %s", str(e), exc_info=True)