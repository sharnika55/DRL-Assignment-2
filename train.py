from envs.stochastic_lander import is_valid_safe_landing


def evaluate_agent(agent, eval_env, num_episodes=100):
  """Evaluates the agent using the centralized safe-landing criteria."""
  successes = 0

  for _ in range(num_episodes):
    state, _ = eval_env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
      action = agent.select_action(state, evaluate=True)
      next_state, reward, terminated, truncated, info = eval_env.step(action)
      state = next_state

      if is_valid_safe_landing(state, terminated, truncated):
        successes += 1
        break

  success_rate = (successes / num_episodes) * 100.0
  return success_rate