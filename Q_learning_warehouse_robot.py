import numpy as np

## Scenario - Robots in a Warehouse
# A growing e-commerce company is building a new warehouse, and the company would like all of the picking operations in the new warehouse to be performed by warehouse robots.
# * In the context of e-commerce warehousing, “picking” is the task of gathering individual items from various locations in the warehouse in order to fulfill customer orders.

# After picking items from the shelves, the robots must bring the items to a specific location within the warehouse where the items can be packaged for shipping.

# In order to ensure maximum efficiency and productivity, the robots will need to learn the shortest path between the item packaging area and all other locations within the warehouse where the robots are allowed to travel.
# * We will use Q-learning to accomplish this task!

# First we made a grid of 11 x 11 to define the environment

environment_rows = 11
environment_columns = 11

# Now we create a 3D numpy array for the environment to hold the current Q-values for each state and action pair Q(s,a)
# The array contains 11 rows and 11 columns, as well as a third "action" dimension
# The "action" dimension consists of 4 values: 0 for up, 1 for right, 2 for down and 3 for left

q_values = np.zeros((environment_rows, environment_columns, 4)) # All the q-values are initialized to 0
actions = ['up', 'right', 'down', 'left']

# We make a 2D array to hold the rewards for the environment. The reward values are: 100 for the packaging area, -100 for obstacles, and -1 for other movable areas
rewards = np.full((environment_rows, environment_columns), -100) # Initializing all the reward values as -100 as most areas are blocked
rewards[0,5] = 100 # Setting up the reward for the packaging area

# We define the aisle for the items of the warehouse
aisles = {} # empty dictionary to store the rewards
aisles[1] = [i for i in range(1,10)] # the selve in the location (1,1) to (1,9)
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

# Set the rewards for all aisles locations
for row_index in range(1,10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = - 1

# Printing the rewards matrix
# for row in rewards:
#     print(row)


# Define a function to specify if the current location is the terminal state
def is_terminal_state(current_row_ind, current_column_ind):
    if rewards[current_row_ind, current_column_ind] == - 1:
        return False
    else:
        return True
    
# Define a function to choose a random non-terminal starting location
def get_random_starting_location():
    current_row_ind = np.random.randint(environment_rows)
    current_column_ind = np.random.randint(environment_columns)

    # Continue choosing random row and column indexes until a non-terminal state is found
    while is_terminal_state(current_row_ind, current_column_ind):
        current_row_ind = np.random.randint(environment_rows)
        current_column_ind = np.random.randint(environment_columns)
    
    return current_row_ind, current_column_ind

# Define an epsilon greedy algorithm that will choose which action to take next 
def get_next_action(current_row_ind, current_column_ind, epsilon):
    if np.random.random() < epsilon: # checking if a random value between 0 and 1 is less than epsilon
        return np.argmax(q_values[current_row_ind, current_column_ind]) # Choose the action with maximum potential rewards
    else:
        return np.random.randint(4) # choose a random action from up, right, down, left

# Define a function that will get the next location based on the chosen action
def get_next_location(current_row_ind, current_column_ind, action_ind):
    new_row_ind, new_column_ind = current_row_ind, current_column_ind
    if actions[action_ind] == 'up' and current_row_ind > 0:
        new_row_ind -= 1
    elif actions[action_ind] == 'right' and current_column_ind < environment_columns - 1:
        new_column_ind += 1
    elif actions[action_ind] == 'down' and current_row_ind < environment_rows - 1:
        new_row_ind += 1
    elif actions[action_ind] == 'left' and current_column_ind > 0:
        new_column_ind -= 1
    return new_row_ind, new_column_ind

# Define a function that will get the shortest path between any location within the warehouse that the robot is allowed to travel
def get_shortest_path(start_row_ind, start_column_ind):
    shortest_path = []
    # return empty path if it is an invalid location
    if is_terminal_state(start_row_ind, start_column_ind):
        return shortest_path
    else: # If it is a legal starting location
        current_row_ind, current_column_ind = start_row_ind, start_column_ind
        shortest_path.append([current_row_ind, current_column_ind])

        # Continue moving along the path until we reach the item packaging location
        while not is_terminal_state(current_row_ind, current_column_ind):
            action_ind = get_next_action(current_row_ind, current_column_ind, 1) # Get the best action with epsilon = 1
            # Move to the next location and add it to the shortest path
            current_row_ind, current_column_ind = get_next_location(current_row_ind, current_column_ind, action_ind)
            shortest_path.append([current_row_ind, current_column_ind])
        return shortest_path
    

# Train the AI agent using Q-Learning
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn

#run through 1000 training episodes
for episode in range(1000):
  #get the starting location for this episode
  row_index, column_index = get_random_starting_location()

  #continue taking actions (i.e., moving) until we reach a terminal state
  #(i.e., until we reach the item packaging area or crash into an item storage location)
  while not is_terminal_state(row_index, column_index):
    #choose which action to take (i.e., where to move next)
    action_index = get_next_action(row_index, column_index, epsilon)

    #perform the chosen action, and transition to the next state (i.e., move to the next location)
    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    #receive the reward for moving to the new state, and calculate the temporal difference
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    #update the Q-value for the previous state and action pair
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')


#display a few shortest paths
print(get_shortest_path(3, 9)) #starting at row 3, column 9
print(get_shortest_path(5, 0)) #starting at row 5, column 0
print(get_shortest_path(9, 5)) #starting at row 9, column 5

