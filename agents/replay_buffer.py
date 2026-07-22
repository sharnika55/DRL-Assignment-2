# =====================================================================
# agents/replay_buffer.py
# =====================================================================
import random
import numpy as np
from collections import deque
from utils.logger import setup_logger

logger = setup_logger("ReplayBuffer")

class ReplayBuffer:
    """
    Fixed-size experience replay buffer for off-policy transition storage and sampling.
    """
    def __init__(self, capacity: int):
        try:
            self.buffer = deque(maxlen=capacity)
            logger.info("Initialized ReplayBuffer with capacity=%d", capacity)
        except Exception as e:
            logger.error("Failed to initialize ReplayBuffer: %s", str(e))
            raise

    def push(self, state, action, reward, next_state, done):
        try:
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.error("Error pushing experience to buffer: %s", str(e))

    def sample(self, batch_size: int):
        try:
            assert len(self.buffer) >= batch_size, "Buffer does not contain enough elements for requested batch size."
            state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
            return (
                np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32)
            )
        except Exception as e:
            logger.error("Error sampling batch from replay buffer: %s", str(e))
            raise

    def __len__(self) -> int:
        return len(self.buffer)