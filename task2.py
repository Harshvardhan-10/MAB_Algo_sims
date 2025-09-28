import numpy as np
from typing import List, Optional, Dict, Tuple
import math

from task1 import kl_divergence

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game ends when any door's health < 0.
    """
    def __init__(self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = np.any(self.health < 0.0)
        return reward, done, {"reward": reward, "health": self.health.copy(), "t": self.t}


# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) to return an int in [0, K-1] to choose an arm.
    - Optionally override update(...) for custom learning.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums   = np.zeros(K, dtype=float)

    def reset_stats(self):
        self.counts[:] = 0
        self.sums[:]   = 0.0

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sums[arm]   += reward

    @property
    def means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)

    def select_arm(self, t: int) -> int:
        raise NotImplementedError

## TASK 2: Make changes here to implement your policy ###
class StudentPolicy(Policy):
    """
    Implement your own algorithm here.
    Replace select_arm with your strategy.
    Currently it has a simple implementation of the epsilon greedy strategy.
    Change this to implement your algorithm for the problem.
    """

    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        super().__init__(K, rng)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(K)
        self.rewards = np.zeros(K)
        self.total_counts = 0
        self.c = 3 # RHS param for KL eqn
        self.health = np.full(K, 100, dtype=float)
        # END EDITING HERE
    
    def select_arm(self, t: int):
        # START EDITING HERE
        for arm in range(self.K):
            if self.counts[arm] == 0:
                return arm
        
        kl_ucb_values = np.zeros(self.K)
        avg_reward = self.rewards/self.counts
        
        for arm in range(self.K):
            val = (math.log(self.total_counts) + self.c * math.log(math.log(max(self.total_counts,math.e)))) / self.counts[arm]
            kl_ucb_values[arm] = self.find_kl_ucb(avg_reward[arm], val, upper_bound=10, lower_bound=avg_reward[arm], precision = 1e-6)

        # T = self.health/kl_ucb_values
    
        return np.argmin(self.health - kl_ucb_values)
        # return np.argmin(T)
        # END EDITING HERE
    
    def update(self, arm_index, reward, health):
        # START EDITING HERE
        self.total_counts += 1
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward
        self.health[arm_index] -= reward
        # END EDITING HERE
    
    def kl_divergence(self, p, q):
        if p == 0:
            return q
        if q == 0:
            return float('inf')
        else:
            return q-p + p*math.log(p/q)

    def find_kl_ucb(self, p, val, upper_bound=3, lower_bound=0, precision=1e-6):
        if val <= 0:
            return p
        while upper_bound - lower_bound > precision:
            mid = (upper_bound + lower_bound) / 2
            if self.kl_divergence(p, mid) > val:
                upper_bound = mid
            else:
                lower_bound = mid
        return (upper_bound + lower_bound) / 2

    # def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
    #     super().__init__(K, rng)
    #     # self.eps = 0.1
    #     # self.health = np.full(K, 100, dtype=float)
    #     self.health = []
    #     self.c = 7.43  # exploration parameter
    #     # self.c = 0.5
    #     self.total_counts = 0

    # def select_arm(self, t: int) -> int:
    #     for door in range(self.K):
    #         if self.counts[door] == 0:
    #             return door

    #     ucb_values = np.zeros(self.K)
    #     avg_reward = self.sums / np.maximum(self.counts, 1)
    #     conf = np.sqrt((self.c * np.log(self.total_counts)) / self.counts)
    #     ucb_values = avg_reward + conf
    #     return np.argmin(self.health - ucb_values)

    # def update(self, arm: int, reward: float, health):
    #     super().update(arm, reward)
    #     self.total_counts += 1
    #     self.health = health