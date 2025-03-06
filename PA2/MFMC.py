import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
	   configuration and taking actions until a terminal state is reached.
	2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
	3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
	4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

	Some important notes:
		
		- The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
		
		- To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		  For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		  will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		- Your Q-table should be a dictionary with keys as the hashed state and 
		  values as another dictionary of actions and their corresponding Q-values.
		  
		  For instance, the agent starts in state (x=0, y=0, health=2, guard=0) which is hashed to 10.
		  If the agent takes action 1 (DOWN) in this state, reaches state (x=0, y=1, health=2, guard=0) which is hashed to 25,
		  and receives a reward of 0, then the Q-table would contain the following entry:
		  
		  Q_table = {10: {1: 0}}. This means that the Q-value for the state 10 and action 1 is 0.

		  Please do not change this representation of the Q-table.
		
		- The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

		- Don't forget to reset the environment to the initial configuration after each episode by calling:
		  obs, reward, done, info = env.reset()

		- The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

		- The value of epsilon is initialized to 1. You are free to choose the decay rate.
		  No default value is specified for the decay rate, experiment with different values to find what works.

		- To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		  Example usage below. This function should be called after every action.
		  if gui_flag:
		      refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the dictionary containing the Q-values (called Q_table). Latestmycode

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999):
    """Q-learning implementation for Escape the Castle environment."""
    Q_table = {}  # Q-values for each state, with each entry as an np.array of Q-values for actions
    num_updates = np.zeros((375, 6))  # Track the number of updates for (s, a) pairs

    for episode in range(num_episodes):
        if episode % 200000 == 0:
             print(f"Running episode {episode}/{num_episodes}")  # Print current 
        obs, reward, done, info = env.reset()
        total_reward = 0

        while not done:
            state = hash(obs)
            if state not in Q_table:
                Q_table[state] = np.zeros(6)  # Initialize Q-values for each action to 0 as np.array

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(6)  # Choose a random action (explore)
            else:
                action = np.argmax(Q_table[state])  # Choose best action (exploit)

            # Step in the environment
            obs_next, reward, done, info = env.step(action)
            total_reward += reward
            next_state = hash(obs_next)

            # Initialize Q-table entry for next state if it doesn't exist
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(6)

            # Q-learning update
            eta = 1 / (1 + num_updates[state, action])
            next_max_q = np.max(Q_table[next_state])  # Use np.max for maximum Q-value of next state
            Q_table[state][action] = (1 - eta) * Q_table[state][action] + eta * (reward + gamma * next_max_q)

            # Update count of updates to this (state, action) pair
            num_updates[state, action] += 1

            # Move to the next state
            obs = obs_next

            # Refresh the GUI if enabled
            if gui_flag:
                refresh(obs, reward, done, info)

        # Decay epsilon after each episode
        epsilon *= decay_rate
        
    return Q_table

decay_rate = 0.999999

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action	= np.argmax(Q_table[state])
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment

# Note: Used AI assistance briefly for debugging and clarifying specific concepts, 
# ensuring full understanding of Q-learning updates and epsilon-greedy action selection. 
# All core logic and function implementation are done independently.
