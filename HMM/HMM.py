import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba


# ### Setup Environment

def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT

def setup_environment(seed=111):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)
    return GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS


# ### Sensor Movements

def simulate_roomba(T, movement_policy,sigma):
    """
    Simulate the movement of a Roomba robot for T time steps and generate noisy observations.

    Parameters:
    - T (int): The number of time steps for which to simulate the Roomba's movement.
    - movement_policy (str): The movement policy dictating how the Roomba moves.
                             Options may include 'straight_until_obstacle' or 'random_walk'.
    - sigma (float): The standard deviation of the Gaussian noise added to the true position 
                     to generate noisy observations.

    Returns:
    - tuple: A tuple containing three elements:
        1. true_positions (list of tuples): A list of the true positions of the Roomba 
                                            at each time step as (x, y) coordinates.
        2. headings (list): A list of headings of the Roomba at each time step.
        3. observations (list of tuples): A list of observed positions with added Gaussian noise,
                                          each as (obs_x, obs_y).
    """
    # Start at the center
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS,is_obstacle,start_pos, start_heading, movement_policy)

    true_positions = []
    observations = []
    headings = []

    print(f"Simulating Roomba movement for policy: {movement_policy}")
    for _ in tqdm(range(T), desc="Simulating Movement"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations


# ### Implement Functions

def emission_probability(state, observation, sigma):
    """
    Calculate the emission probability in log form for a given state and observation using a Gaussian distribution.

    Parameters:
    - state (tuple): The current state represented as (position, heading), 
                     where position is a tuple of (x, y) coordinates.
    - observation (tuple): The observed position as a tuple (obs_x, obs_y).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - float: The log probability of observing the given observation from the specified state.
    """
    true_position = state[0]
    true_x, true_y = true_position
    obs_x, obs_y = observation
    
    diff_x = obs_x - true_x
    diff_y = obs_y - true_y
    
    squared_distance = diff_x**2 + diff_y**2
    
    log_probability = -0.5 * (squared_distance / (sigma**2)) - np.log(np.sqrt(2 * np.pi * sigma**2))
    
    return log_probability

def straight_until_obstacle_possible_states(prev_state):
    """
    Find all possible new states that can be reached from the previous state using the 'straight_until_obstacle' policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                        where position is a tuple of (x, y) coordinates and heading is a direction.

    Returns:
    - list of tuples: A list of all possible new states that can be reached from the current state.
    """
    prev_position, prev_heading = prev_state
    possible_states = []

    dx,dy = MOVEMENTS[prev_heading]
    new_position = (prev_position[0] + dx, prev_position[1] + dy)

    if is_obstacle(new_position):
        for heading in HEADINGS:
            prev_heading = heading
            dx, dy = MOVEMENTS[prev_heading]
            new_position = (prev_position[0] + dx, prev_position[1] + dy)
            if is_obstacle(new_position):
                possible_states.append((prev_position, prev_heading))
            else :
                possible_states.append((new_position, prev_heading))
    else:
        possible_states.append((new_position, prev_heading))

    return possible_states

def random_walk_possible_states(prev_state):
    """
    Find all possible new states that can be reached from the previous state using the 'random_walk' policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                        where position is a tuple of (x, y) coordinates and heading is a direction.

    Returns:
    - list of tuples: A list of all possible new states that can be reached from the previous state.
    """
    prev_position, prev_heading = prev_state
    possible_states = []

    for heading in HEADINGS:
        prev_heading = heading
        dx, dy = MOVEMENTS[prev_heading]
        new_position = (prev_position[0] + dx, prev_position[1] + dy)
        if is_obstacle(new_position):
            possible_states.append((prev_position, prev_heading))
        else :
            possible_states.append((new_position, prev_heading))

    return possible_states

def transition_probability(prev_state, curr_state, movement_policy):
    """
    Calculate the transition probability in log form between two states based on a given movement policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                        where position is a tuple of (x, y) coordinates and heading is a direction.
    - curr_state (tuple): The current state represented as (position, heading),
                        similar to prev_state.
    - movement_policy (str): The movement policy that dictates how transitions are made. 
                            Options are 'straight_until_obstacle' and 'random_walk'.

    Returns:
    - float: The log probability of transitioning from prev_state to curr_state given the movement policy.
            Returns 0.0 (log(1)) for certain transitions, -inf (log(0)) for impossible transitions,
            and a uniform log probability for equal transitions in the case of random walk.
    """
    ###### SENSON MARKOV ASSUMPTION ######

    if is_obstacle(prev_state[0]):
        return float('-inf')

    if movement_policy == 'straight_until_obstacle':
        prev_position, prev_heading = prev_state
        possible_states = []

        dx, dy = MOVEMENTS[prev_heading]
        new_position = (prev_position[0] + dx, prev_position[1] + dy)

        if is_obstacle(new_position):
            heading_index = 0
            while heading_index < len(HEADINGS):
                prev_heading = HEADINGS[heading_index]
                dx, dy = MOVEMENTS[prev_heading]
                new_position = (prev_position[0] + dx, prev_position[1] + dy)
                if is_obstacle(new_position):
                    possible_states.append((prev_position, prev_heading))
                else:
                    possible_states.append((new_position, prev_heading))
                heading_index += 1
        else:
            possible_states.append((new_position, prev_heading))

    elif movement_policy == 'random_walk':
        prev_position, prev_heading = prev_state
        possible_states = []

        heading_index = 0
        while heading_index < len(HEADINGS):
            prev_heading = HEADINGS[heading_index]
            dx, dy = MOVEMENTS[prev_heading]
            new_position = (prev_position[0] + dx, prev_position[1] + dy)
            if is_obstacle(new_position):
                possible_states.append((prev_position, prev_heading))
            else:
                possible_states.append((new_position, prev_heading))
            heading_index += 1

    else:
        raise ValueError("Unknown movement policy")

    if curr_state in possible_states:
        prob = 1 / len(possible_states)
        return np.log(prob)
    else:
        return float('-inf')

# ### Viterbi Algorithm
def viterbi(observations, start_state, movement_policy, states, sigma):
    """
    Perform the Viterbi algorithm to find the most likely sequence of states given a series of observations.

    Parameters:
    - observations (list of tuples): A list of observed positions, each as a tuple (obs_x, obs_y).
    - start_state (tuple): The initial state represented as (position, heading),
                           where position is a tuple of (x, y) coordinates.
    - movement_policy (str): The movement policy that dictates how transitions are made.
                             Options are 'straight_until_obstacle' and 'random_walk'.
    - states (list of tuples): A list of all possible states, each represented as (position, heading).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - list of tuples: The most probable sequence of states that could have led to the given observations.
    """
    if movement_policy == "random_walk":
        T = len(observations)
        prev_V = {state: emission_probability(state, observations[1], sigma) + transition_probability(start_state, state, movement_policy) for state in states}
        prev_path = {state: [start_state, state] for state in states}
        t = 2
        while t < T:
            V_t = {}
            path_t = {}
            curr_state_idx = 0
            while curr_state_idx < len(states):
                curr_state = states[curr_state_idx]
                max_prob, max_state = max(
                    (prev_V[prev_state] +
                    transition_probability(prev_state, curr_state, movement_policy) +
                    emission_probability(curr_state, observations[t], sigma), prev_state)
                    for prev_state in states
                )
                V_t[curr_state] = max_prob
                path_t[curr_state] = prev_path[max_state] + [curr_state]
                curr_state_idx += 1
            prev_V = V_t
            prev_path = path_t
            t += 1

        max_final_state = max(prev_V, key=prev_V.get)
        return prev_path[max_final_state]
    else:
        num_observations = len(observations)
        num_states = len(states)
        viterbi_table = np.full((num_states, num_observations), -np.inf)
        backpointer = np.zeros((num_states, num_observations), dtype=int)
        
        i = 0
        while i < len(states):
            emission_prob = emission_probability(states[i], observations[0], sigma)
            transition_prob = transition_probability(start_state, states[i], movement_policy)
            viterbi_table[i, 0] = transition_prob + emission_prob
            i += 1
        
        t = 1
        while t < num_observations:
            curr_state_idx = 0
            while curr_state_idx < len(states):
                max_prob = -np.inf
                max_state_idx = -1
                prev_state_idx = 0
                while prev_state_idx < len(states):
                    transition_prob = transition_probability(states[prev_state_idx], states[curr_state_idx], movement_policy)
                    emission_prob = emission_probability(states[curr_state_idx], observations[t], sigma)
                    prob = viterbi_table[prev_state_idx, t-1] + transition_prob + emission_prob
                    if prob > max_prob:
                        max_prob = prob
                        max_state_idx = prev_state_idx
                    prev_state_idx += 1
                viterbi_table[curr_state_idx, t] = max_prob
                backpointer[curr_state_idx, t] = max_state_idx
                curr_state_idx += 1
            t += 1

        best_last_state_idx = np.argmax(viterbi_table[:, num_observations - 1])
        best_path = [states[best_last_state_idx]]
        t = num_observations - 1
        while t > 0:
            best_last_state_idx = backpointer[best_last_state_idx, t]
            best_path.insert(0, states[best_last_state_idx])
            t -= 1
        
        return best_path
      
# ### Evaluation (DO NOT CHANGE THIS)
def getestimatedPath(policy, results, states, sigma):
    """
    Estimate the path of the Roomba using the Viterbi algorithm for a specified policy.

    Parameters:
    - policy (str): The movement policy used during simulation, such as 'random_walk' or 'straight_until_obstacle'.
    - results (dict): A dictionary containing simulation results for different policies. Each policy's data includes:
                      - 'true_positions': List of true positions of the Roomba at each time step.
                      - 'headings': List of headings of the Roomba at each time step.
                      - 'observations': List of noisy observations at each time step.
    - states (list of tuples): A list of all possible states (position, heading) for the Hidden Markov Model.
    - sigma (float): The standard deviation of the Gaussian noise used in the emission probability.

    Returns:
    - tuple: A tuple containing:
        1. true_positions (list of tuples): The list of true positions from the simulation.
        2. estimated_path (list of tuples): The most likely sequence of states estimated by the Viterbi algorithm.
    """
    print(f"\nProcessing policy: {policy}")
    data = results[policy]
    observations = data['observations']
    start_state = (data['true_positions'][0], data['headings'][0])
    estimated_path = viterbi(observations, start_state, policy, states, sigma)
    return data['true_positions'], estimated_path


def evaluate_viterbi(estimated_path, true_positions, T,policy):
    """
    Evaluate the accuracy of the Viterbi algorithm's estimated path compared to the true path.
    """
    correct = 0
    for true_pos, est_state in zip(true_positions, estimated_path):
        if true_pos == est_state[0]:
            correct += 1
    accuracy = correct / T * 100
    # data['accuracy'] = accuracy
    print(f"Tracking accuracy for {policy.replace('_', ' ')} policy: {accuracy:.2f}%")
111

def plot_results(true_positions, observations, estimated_path, policy):
    """
    Plot the true and estimated paths of the Roomba along with the noisy observations.
    The function plots and saves the graphs of the true and estimated paths.
    """
    # Extract coordinates
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    obs_x = [obs[0] for obs in observations]
    obs_y = [obs[1] for obs in observations]
    est_x = [state[0][0] for state in estimated_path]
    est_y = [state[0][1] for state in estimated_path]

    # Identify start and end positions
    start_true = true_positions[0]
    end_true = true_positions[-1]
    start_est = estimated_path[0][0]
    end_est = estimated_path[-1][0]

    # Plotting
    plt.figure(figsize=(10, 10))

    # True Path Plot
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the true path
    plt.scatter(*start_true, c='b', marker='o', s=100, label='True Start', edgecolors='black')
    plt.scatter(*end_true, c='purple', marker='X', s=100, label='True End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - True Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    # Estimated Path Plot
    plt.subplot(2, 1, 2)
    plt.plot(est_x, est_y, 'b--', label='Estimated Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the estimated path
    plt.scatter(*start_est, c='b', marker='o', s=100, label='Estimated Start', edgecolors='black')
    plt.scatter(*end_est, c='purple', marker='X', s=100, label='Estimated End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - Estimated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    fname = f"{policy.replace('_', ' ')}_Policy_Roomba_Path_Tracking.png"
    plt.savefig(fname)

# import csv

# def save_estimated_paths_to_csv(estimated_paths, filename="estimated_paths_220.csv"):
#     """
#     Save the estimated paths to a CSV file.

#     Parameters:
#     - estimated_paths (dict): A dictionary containing the estimated paths for each policy.
#                               Each key is a policy, and the value is the estimated path.
#     - filename (str): The name of the CSV file to save the estimated paths.
#     """
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Policy", "Time Step", "X", "Y", "Heading"])

#         for policy, path in estimated_paths.items():
#             for t, state in enumerate(path):
#                 position, heading = state
#                 x, y = position
#                 writer.writerow([policy, t, x, y, heading])

if __name__ == "__main__":
    # 1. Set up the environment, including grid size, headings, and movements.
    seed = 111
    # print("seed=",i)
    setup_environment(seed)
    sigma = 1.0  # Observation noise standard deviation
    T = 50       # Number of time steps

    # Simulate for both movement policies
    policies = ['random_walk', 'straight_until_obstacle']
    results = {}

    # 2. Loop through each movement policy and simulate the Roomba's movement:
    #    - Generate true positions, headings, and noisy observations.
    #    - Store the results in the dictionary.
    for policy in policies:
        true_positions, headings, observations = simulate_roomba(T, policy,sigma)
        results[policy] = {
            'true_positions': true_positions,
            'headings': headings,
            'observations': observations
        }

    # 3. Define the HMM components
    #   - A list (states) containing all possible states of the Roomba, where each state is represented as a tuple ((x, y), h)
    #   - x, y: The position on the grid.
    #   - h: The heading or direction (e.g., 'N', 'E', 'S', 'W').
    states = []
    ###### YOUR CODE HERE ######
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            for h in HEADINGS:
                states.append(((x, y), h))
    
    # 4. Loop through each policy to estimate the Roomba's path using the Viterbi algorithm:
    #    - Retrieve the true positions and estimated path.
    #    - Evaluate the accuracy of the Viterbi algorithm.
    #    - Plot the true and estimated paths along with the observations.
    # estimated_paths = {}

    for policy in policies:
        true_positions, estimated_path = getestimatedPath(policy,results,states,sigma)
        evaluate_viterbi(estimated_path, true_positions, T,policy)
        plot_results(true_positions, observations, estimated_path, policy)
        # estimated_paths[policy] = estimated_path

    # save_estimated_paths_to_csv(estimated_paths)