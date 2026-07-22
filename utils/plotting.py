# =====================================================================
# utils/plotting.py
# =====================================================================
import os
import matplotlib.pyplot as plt
from config import Config
from utils.logger import setup_logger

logger = setup_logger("Plotting")

def plot_performance_results(results: dict):
    """
    Generates and saves the 4 required performance evaluation plots.
    """
    logger.info("Generating performance evaluation comparison plots...")
    try:
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(Config.PLOT_DIR, Config.PLOT_FILENAME)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Episode Reward vs Training Episode
        for key, res in results.items():
            axs[0, 0].plot(res["rewards"], label=key, alpha=0.7)
        axs[0, 0].set_title("Episode Reward vs Training Episode")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2: Average Predicted Q-Value vs Training Episode
        for key, res in results.items():
            axs[0, 1].plot(res["q_values"], label=key, alpha=0.7)
        axs[0, 1].set_title("Average Predicted Q-Value vs Training Episode")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Mean Q-Value (Validation Set)")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot 3: Successful Landing Rate (Moving Average over 100 episodes)
        for key, res in results.items():
            axs[1, 0].plot(res["success_rate"], label=key, alpha=0.8)
        axs[1, 0].set_title("Successful Landing Rate (Moving Avg 100 Ep)")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Success Rate")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot 4: Average Number of Thruster Activations per Episode
        for key, res in results.items():
            axs[1, 1].plot(res["thruster_counts"], label=key, alpha=0.7)
        axs[1, 1].set_title("Thruster Activations per Episode")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Thruster Count")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info("Saved performance comparison plot to '%s'", plot_path)
        plt.show()
    except Exception as e:
        logger.error("Error generating plots: %s", str(e))
        raise