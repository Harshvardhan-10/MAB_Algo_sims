# Multi-Armed Bandit Assignment

## Problem Statements

### Task 1 (4 marks)

In this first task, you will implement the sampling algorithms: (1) UCB, (2) KL-UCB, and (3) Thompson Sampling. This task is straightforward based on the class lectures. The instructions below tell you about the code you are expected to write.

Read `task1.py`. It contains a sample implementation of epsilon-greedy for you to understand the Algorithm class. You have to edit the `__init__` function to create any state variables your algorithm needs, and implement `give_pull` and `get_reward`. 

- `give_pull` is called whenever it's the algorithm's decision to pick an arm and it must return the index of the arm your algorithm wishes to pull (lying in 0, 1, ... self.num_arms-1). 
- `get_reward` is called whenever the environment wants to give feedback to the algorithm. It will be provided the `arm_index` of the arm and the `reward` seen (0/1). 

Note that the `arm_index` will be the same as the one returned by the `give_pull` function called earlier. For more clarity, refer to `single_sim` function in `simulator.py`.

Once done with the implementations, you can run `simulator.py` to see the regrets over different horizons. Save the generated plot and add it your report, with apt captioning. In your report, also describe any details which are relevant to your particular implementation, such as parameter values you have chosen/tuned, or a certain way in which you have coded some function. You may also run `autograder.py` to evaluate your algorithms on the provided test instances.

### Task 2 (4 marks)

In this task you will interact with a stochastic environment.

#### Problem Statement

You are in a room with multiple doors, and you must break through at least one door in order to move to the other side. Each door begins with a strength of 100. At every step, you may choose one door to strike. The damage dealt to a door is drawn from a Poisson distribution, with the mean value specific to that door. The door's remaining strength is then reduced by the realised damage.

For example, if the current strength of a door is *s* and its associated mean is *μ*, and you choose to strike this door, then the new strength is *s - d*, where *d ~ Poisson(μ)*. The game terminates immediately once the strength of any door drops below zero.

A problem instance for this task is defined by an array of *n* positive real numbers, where *n* is the number of doors and each element represents the mean of the Poisson distribution for that door. The means of the doors have values in (0, 3) (that is, positive values less than 3), and the number of doors is at most 30. Your objective is to design a strategy that breaks through a door in the minimum expected number of strikes.

#### Objective

Develop a selection policy that minimises the expected number of strikes (steps) required for the episode to terminate. In other words, your policy should efficiently identify and focus on doors with higher expected damage, while accounting for the fact that the game ends as soon as any door's strength falls below zero.

#### Code Provided

- **`task2.py`**: Contains the implementation of the Poisson environment. You must implement your algorithm by modifying the `StudentPolicy` class. By default, it contains a simple Epsilon-Greedy implementation.
- **`door_game.py`**: An interactive simulator where you can play the game manually or run your algorithm to visualise its behaviour. This file is provided only for exploration and analysis; it is not used for grading.
- **`simulator.py`**: Runs simulations of your policy in the environment. Need not be modified. To test on custom testcases make changes in the main function.

#### Evaluation

Your algorithm will be evaluated using the autograder. Each test case specifies a threshold for the maximum number of strikes allowed. For evaluation, the average number of strikes over 200 episodes is measured. Your policy must achieve an average below the threshold to pass the test case.

All test cases must complete execution within 5 minutes. This limit is intentionally generous; it ensures that algorithms do not run excessively long.

For Task 2, you only need to edit `task2.py`. The `door_game.py` file is provided exclusively for visualisation and debugging of your policy.

#### Hint

Read the Wikipedia page on Poisson distribution for insights that may help in designing your algorithm. You are also encouraged to explore the academic literature. Be sure to cite any references you have consulted in `references.txt` (see under "Submission").

### Task 3 (4 marks)

In this task you will optimise the computations of the KL-UCB algorithm for multi-armed bandits. The goal is to reduce the computational overhead of KL-UCB while maintaining good regret performance. You will implement an optimised version of KL-UCB that achieves significant speedup.

For this task, you must edit the file `task3.py`. You need to implement one algorithm class:

**`KL_UCB_Optimized`**: An optimised version that reduces computational cost while maintaining regret performance within acceptable thresholds.

#### Key Requirements:

- The `KL_UCB_Optimized` algorithm must achieve regret performance within the thresholds defined in the testcases folder.
- The optimised algorithm must achieve the threshold speedups defined in the testcases folder, over the standard KL_UCB.
- The algorithm implemented in this task must be some version of kl ucb only (i.e. some modification over the original kl-ucb algorithm) and not any other algorithm like Thompson Sampling. In other words, it must involve selecting actions greedily with respect to the KL upper confidence bound. If any algorithm other than kl-ucb is implemented, then 0 marks will be given for task 3.

#### Hint For Optimisation

In a typical run of KL-UCB, there will be long sequences in which the same arm gets pulled, before switching to another one. On a given decision-making step, can you decide to give an arm many pulls (say m >= 2) without consulting the data and re-calculating UCBs while performing these m pulls?

#### Theoretical Question and Bonus Problem

Suppose we run a deterministic algorithm, and also fix the random seed to generate rewards from the bandit instance. Then on each run, we will get the same action-reward sequence a₀, r₀, a₁, r₁, a₂, r₂, .... Can you think of a computational optimisation of KL-UCB such that both algorithms (original and optimised) are guaranteed to produce the same action-reward sequence when the seed is fixed? In other words, if the run of KL-UCB picks arms 1, 3, 4, 2, 1 in the first five pulls, and gets rewards of 0, 1, 1, 0, 1, so must your optimised variant of KL-UCB, which will typically run faster. In turn, this will mean that both algorithms have the same cumulative regret vs. horizon graph (except possibly for numerical differences on account of floating point operations). How would you implement such an algorithm? Answer this question in your report. This question is for 1 mark. 

In `task3.py`, a bonus problem is also given, if you have the answer to this optimisation problem, you can implement it in the `KL_UCB_Optimized_Bonus` class in `task3.py`. This is an optional problem and will not affect your marks for task 3 if you do not implement it.

You only need to implement the algorithm classes in `task3.py` and make changes to the code in `task3.py` only. For simulation, you can use the `simulator.py` function.

## Report

### For Task 1
Your report needs to have all the plots (UCB, KL-UCB, Thompson) that `simulator.py` generates (uncomment the final lines). There are 3 plots in total for task 1. You may, of course, include any additional plots you generate. Your plots should be neatly labelled and captioned for the report. Also explain your code for the three algorithms.

### For Task 2
Your report should present a clear explanation of your overall approach for the problem with Poisson Doors, describe in detail the policy you implemented and the reasoning behind its design, and provide a well-argued justification of why your method achieves good performance in this setting, both in terms of theoretical soundness and practical effectiveness.

### For Task 3
You need to explain your optimisation approach, provide performance analysis comparing standard vs optimised algorithms, and discuss the techniques used to achieve speedup while maintaining regret optimality. You need to include the plots in the report which are generated after running the `simulator.py` for task3.
