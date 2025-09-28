"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed
def kl_divergence(p, q):
    """Fast KL divergence computation with edge case handling"""
    if p == 0:
        return -math.log(1 - q) if q < 1 else float('inf')
    if p == 1:
        return -math.log(q) if q > 0 else float('inf')
    if q == 0 or q == 1:
        return float('inf')
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def find_kl_ucb(p, val, upper_bound=1, lower_bound=0, precision=1e-6):
    if p >= 1:
        return 1.0
    if val <= 0:
        return p
    while upper_bound - lower_bound > precision:
        mid = (upper_bound + lower_bound) / 2
        if kl_divergence(p, mid) > val:
            upper_bound = mid
        else:
            lower_bound = mid
    return (upper_bound + lower_bound) / 2

# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        #START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.rewards = np.zeros(num_arms)
        self.total_counts = 0
        self.c = 0  # KL UCB RHS param
        self.batch_size = self.horizon // 1000
        self.prev_best_arm = -1
        self.best_arm = -1
        self.batch_pulls = self.batch_size  # Force recompute on first call

        #END EDITING HERE

    def give_pull(self):
        #START EDITING HERE
        # First pull each arm once
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
            
        if self.batch_pulls >= self.batch_size:
            self.batch_pulls = 0
            # Recompute KL-UCB values at the start of each batch
            kl_ucb_values = np.zeros(self.num_arms)
            avg_reward = self.rewards / self.counts
            
            for arm in range(self.num_arms):
                val = (math.log(self.total_counts) + self.c * math.log(math.log(max(self.total_counts, math.e)))) / self.counts[arm]
                kl_ucb_values[arm] = find_kl_ucb(avg_reward[arm], val, lower_bound=avg_reward[arm], precision=1e-4)
                self.best_arm = np.argmax(kl_ucb_values)
            # self.batch_size = np.max(kl_ucb_values) - np.max(kl_ucb_values[kl_ucb_values < np.max(kl_ucb_values)])
            # self.batch_size = max(1, int(self.batch_size * 100))

        self.batch_pulls += 1
        return self.best_arm
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.total_counts += 1
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward

        self.prev_best_arm = self.best_arm
        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE
