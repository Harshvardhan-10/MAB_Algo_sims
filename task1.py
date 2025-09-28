"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def kl_divergence(p, q):
    if p == 0:
        return -math.log(1 - q) if q < 1 else float('inf')
    if p == 1:
        return -math.log(q) if q > 0 else float('inf')
    else:
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

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.rewards = np.zeros(num_arms)
        self.total_counts = 0
        self.c = 1.5 # higher than 2 works better for short term
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
            
        ucb_values = np.zeros(self.num_arms)
        avg_reward = self.rewards/self.counts
        conf = np.sqrt((self.c * np.log(self.total_counts)) / self.counts)
        ucb_values = avg_reward + conf
        return np.argmax(ucb_values)
        # END EDITING HERE


    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_counts += 1
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward 
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.rewards = np.zeros(num_arms)
        self.total_counts = 0
        self.c = 0 # exploration parameter
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm
        
        kl_ucb_values = np.zeros(self.num_arms)
        avg_reward = self.rewards/self.counts
        
        for arm in range(self.num_arms):
            val = (math.log(self.total_counts) + self.c * math.log(math.log(max(self.total_counts,math.e)))) / self.counts[arm]
            kl_ucb_values[arm] = find_kl_ucb(avg_reward[arm], val, upper_bound=1, lower_bound=avg_reward[arm], precision=1e-6)

        return np.argmax(kl_ucb_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_counts += 1
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.success = np.zeros(num_arms)
        self.total_counts = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = np.random.beta(self.success + 1, self.counts - self.success + 1)
        return np.argmax(samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_counts += 1
        self.counts[arm_index] += 1
        self.success[arm_index] += reward
        # END EDITING HERE

