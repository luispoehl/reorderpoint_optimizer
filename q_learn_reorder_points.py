import numpy as np

def get_optimal_reorder_point(current_inventory_level,optimal_reorder_points):
    #for state, reorder_point in enumerate(optimal_reorder_points):
    #    print(f"State {state}: Optimal Reorder Point = {reorder_point}")
    #print(f"Overall costs: {overall_costs}")
    return optimal_reorder_points[current_inventory_level]

def optimize(storage_cost_per_day,downtime_cost_per_day,reorder_point,init_inventory,order_amount,error_scale,error_shape,lead_time_range):


    # Define Q-Learning parameters
    num_states = 20  # Number of inventory levels (0 to 100)
    num_actions = 20  # Number of possible reorder points (0 to 100)
    learning_rate = 0.01  # Learning rate
    discount_factor = 0.8  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 100000  # Number of episodes
    initial_reorder = reorder_point

    # Initialize Q-table with small random values
    Q = np.random.rand(num_states, num_actions)

    # Simulate inventory stock to calculate overall costs
    def simulate_one_week(reorder_point):
        #init_inventory=main.init_inventory
        #reorder_point=main.reorder_point
        #order_amount=main.order_amount
        #error_scale=main.error_scale
        #error_shape=main.error_shape
        #lead_time_range=main.lead_time_range
        num_weeks=1

        # Generate random lead times
        lead_times = np.random.uniform(lead_time_range[0], lead_time_range[1], num_weeks)

        # Initialize inventory and reorder flags
        inventory = init_inventory
        reorder_flag = False

        # Initialize lists to store data for visualization
        inventory_levels = []
        reorder_flags = []

        # Simulate inventory stock over time
        remaining_lifetime = np.random.weibull(error_shape) * error_scale
        ordered_weeks_ago = 0
        remaining_lifetime -= 1
        if remaining_lifetime <= 0:
            inventory -= 1
            remaining_lifetime = np.random.weibull(error_shape) * error_scale

        # Check if reorder is needed
        if inventory <= reorder_point and not reorder_flag:
            reorder_flag = True
            ordered_weeks_ago = 0

        # Check if order has arrived
        i=0
        if reorder_flag and ordered_weeks_ago >= lead_times[i]: #i=0 bc. only one week
            reorder_flag = False
            inventory += order_amount

        inventory = max(inventory, 0)
        # Store inventory level and reorder flag for visualization
        inventory_levels.append(inventory)
        reorder_flags.append(reorder_flag)

        if reorder_flag:
            ordered_weeks_ago += 1

        return inventory_levels, reorder_flags


    for episode in range(num_episodes):
        # Reset the environment and obtain initial state
        inventory_levels, reorder_flags = simulate_one_week(initial_reorder)
        state = inventory_levels[0]

        # Calculate accumulated costs
        storage_costs = np.zeros(num_states)
        downtime_costs = np.zeros(num_states)

        for state in range(num_states):
            inventory_levels, _ = simulate_one_week(state)
            storage_costs[state] = sum(max(level, 0) * storage_cost_per_day for level in inventory_levels)
            downtime_costs[state] = sum(downtime_cost_per_day if level <= 0 else 0 for level in inventory_levels)

        for step in range(len(inventory_levels) - 1):
            # Exploration vs. exploitation
            if np.random.uniform() < epsilon:
                # Explore: choose a random action
                action = np.random.randint(num_actions)
            else:
                # Exploit: choose the action with the highest Q-value
                action = np.argmax(Q[state, :])

            # Take the selected action and observe the next state and immediate reward
            next_state = inventory_levels[step + 1]
            reward = -1 * (storage_costs[next_state] + downtime_costs[next_state])

            # Update Q-value of the current state-action pair
            Q[state, action] = Q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            # Transition to the next state
            state = next_state

    # Retrieve the optimal reorder point (action) for each state (inventory level)
    optimal_reorder_points = np.argmax(Q, axis=1)

    # Evaluate the performance of the learned policy
    inventory_levels, reorder_flags = simulate_one_week(optimal_reorder_points[initial_reorder])
    overall_costs = storage_costs[-1] + downtime_costs[-1]

    # Print the optimal reorder points and overall costs
    return(optimal_reorder_points,overall_costs)