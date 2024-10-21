import numpy as np
import random
from racetrack_environment import RaceTrackEnv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.notebook import tqdm
plt.style.use('dark_background')

from utils import build_uturn_course

# Build the course
_course_dim = (8, 10)
_inner_wall_dim = (2, 6)

course = build_uturn_course(_course_dim, _inner_wall_dim)
track = RaceTrackEnv(course)
for row in course:
    print(row)

pos_map = track.course  # overlay track course
#plt.imshow(pos_map, cmap='hot', interpolation='nearest')
#plt.show()


# Select course and initialize dummy policy

course = build_uturn_course(_course_dim, _inner_wall_dim)
track = RaceTrackEnv(course)
dummy_slow_pi = np.ones([track.bounds[0], track.bounds[1], 1+2*track.MAX_VELOCITY, 1+2*track.MAX_VELOCITY]) * 4 

dummy_slow_pi[:track.bounds[0]//2, :, 0 , 0] = 5  # accelerate right
dummy_slow_pi[:track.bounds[0]//2, -2:, 0, :] = 6  # accelerate bottom left
dummy_slow_pi[-2:, track.bounds[1]//2:, :, 0] = 0  # accelerate top left

pi = dummy_slow_pi

# Function templates to help structure the code, these (and similar ones) will be used in future exercises as well
# Note that the environment/track is a global variable here, so it does not need to be passed as an argument
# This is not a good practice in general, but it is done here to simplify the code
# You can therefore use the track variable directly in the functions
# e.g., track.reset()

def interact(pi, state):
    """Interact with the environment to get to the next state.

    Args:
        pi: The policy to follow
        state: The current state before interaction

    Returns:
        next_state: The next state after interaction
        reward: The reward for the current interaction
        terminated: If the goal was reached
        truncated: If the boundary of the track was breached
    """
    
    #import ipdb;ipdb.set_trace()
    action = track.action_to_tuple(pi[*state])    
    next_state, reward, terminated, truncated, _ = track.step(action)

    return next_state, reward, terminated, truncated


def gather_experience(pi, max_epsiode_len):
    """Simulate a full episode of data by repeatedly interacting with the environment.

    End the episode when the finish line is reached. Whenever the car leaves the track, simply
    reset the environment.
    
    Args:
        pi: The policy to apply
        max_episode_len: The number of steps at which the episode is terminated automatically

    Returns:
        states: All states that were visited in the episode
        rewards: All rewards that were acquired in the episode
        visited_states: The unique states that were visited in the episode
        first_visit_list: Whether it was the first time in the episode that the
            state at the same index was visited
    """
    # initialize variables in which collected data will be stored
    states = [] # list of tuples
    rewards = [] # list of floats
    #visited_states = set() # set of tuples
    visited_states = {} # dict of tuples
    first_visit_list = [] # list of booleans

    # reset environment and start episode
    state = track.reset()
    # There will be two different ways to end the episode: reaching the finish line or reaching the maximum number of steps
    # If the car hits the boundary of the track, the environment will be reset and the episode will continue
    # Therefore, you have to handle termination and truncation differently
    counter_visited_states = 0
    for k in range(max_epsiode_len):
        
        # save the momentary state and check for first_visit
        state = [state[0][0], state[0][1], state[1][0], state[1][1]]
        states.append(state)
        #import ipdb;ipdb.set_trace()
        visited = False
        for itr in range(len(visited_states)):
            if state == visited_states[itr]:
                visited == True
        if not visited:
            first_visit_list.append(state)
        #if state not in visited_states:
        #    first_visit_list.append(state)
        #first_visit_list.append(state not in visited_states)
        #visited_states.add(state)
        visited_states[counter_visited_states] = state
        counter_visited_states += 1
        
        # interact with the environment
        next_state, reward, terminated, truncated = interact(pi, state)

        # reset the environment if the car left the track
        if truncated:
            next_state = track.reset()

        # update the state for the next iteration
        state = next_state

        # save received reward
        rewards.append(reward)
        # terminate the environment if the finish line was passed
        if terminated: 
            break 

    return states, rewards, visited_states, first_visit_list

def learn(values, n_dict, states, rewards, first_visit_list, gamma):
    """Learn from the collected data using the incremental first-visit MC-based prediction algorithm.

    Args:
        values: The state-value estimates before the current update step
        n_dict: The state visit counts before the current update step
        states: All states that were visited in the last episode
        rewards: All rewards that were visited in the last episode
        first_visit_list: Whether it was the first time in the episode that the
            state at the same index was visited
        gamma: Discount factor

    Returns:
        values: The updated state-value estimates
        n_dict: The state visit counts after the current update step
    """
    ### BEGIN SOLUTION
    g = 0  
    for state, reward, first_visit in zip(states[::-1], rewards[::-1], first_visit_list[::-1]): # count backwards

        # calculate return
        g = gamma * g + reward
        
        # update values if it is the first visit in the episode
        if first_visit:
            n_dict[*state]=1
            
        # Count visits to this state in n_dict
        n_dict[*state] = n_dict[*state] + 1

        # add new return g to existing value
        values[*state] += 1/n_dict[*state] * (g-values[*state])

    ### END SOLUTION
    return values, n_dict


# initialize the value function
values = np.zeros([track.bounds[0], track.bounds[1], 1+2*track.MAX_VELOCITY, 1+2*track.MAX_VELOCITY])

# initialize an empty dict to count the number of visits
# note that in the lecture the list l(x_k) was used for this purpose
n_dict = {}

# configuration parameters
gamma = 1 # discount factor
no_episodes = 500 # number of evaluated episodes
max_episode_len = 2000 # number of allowed timesteps per episode

for e in tqdm(range(no_episodes), position=0, leave=True):

    states, rewards, visited_states, first_visit_list = gather_experience(pi, max_episode_len)

    #import ipdb;ipdb.set_trace()
    values, n_dict = learn(values, n_dict, states, rewards, first_visit_list, gamma)
    