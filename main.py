import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from q_learn_reorder_points import optimize

def simulate_inventory_stock(init_inventory, reorder_point, order_amount, error_scale, error_shape, lead_time_range, num_weeks):
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
    ordered_weeks_ago = 0  # Weeks since the order was placed
    for i in range(num_weeks):
        remaining_lifetime -= 1
        if remaining_lifetime <= 0:
            inventory -= 1
            remaining_lifetime = np.random.weibull(error_shape) * error_scale

        # Check if reorder is needed
        if inventory <= reorder_point and not reorder_flag:
            reorder_flag = True
            ordered_weeks_ago = 0

        # Check if order has arrived
        if reorder_flag and ordered_weeks_ago >= lead_times[i]:
            reorder_flag = False
            inventory += order_amount

        inventory = max(inventory, 0)
        # Store inventory level and reorder flag for visualization
        inventory_levels.append(inventory)
        reorder_flags.append(reorder_flag)

        if reorder_flag:
            ordered_weeks_ago += 1

    return inventory_levels, reorder_flags

def simulate_optimized_inventory_stock(optimal_reorder_points, init_inventory, order_amount, error_scale, error_shape, lead_time_range, num_weeks):
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
    ordered_weeks_ago = 0  # Weeks since the order was placed
    for i in range(num_weeks):
        try:
            reorder_point = optimal_reorder_points[inventory]
        except:
            reorder_point = reorder_point
        remaining_lifetime -= 1
        if remaining_lifetime <= 0:
            inventory -= 1
            remaining_lifetime = np.random.weibull(error_shape) * error_scale

        # Check if reorder is needed
        if inventory <= reorder_point and not reorder_flag:
            reorder_flag = True
            ordered_weeks_ago = 0

        # Check if order has arrived
        if reorder_flag and ordered_weeks_ago >= lead_times[i]:
            reorder_flag = False
            inventory += order_amount

        inventory = max(inventory, 0)
        # Store inventory level and reorder flag for visualization
        inventory_levels.append(inventory)
        reorder_flags.append(reorder_flag)

        if reorder_flag:
            ordered_weeks_ago += 1

    return inventory_levels, reorder_flags


# Configure collapsible section for general configurations
with st.sidebar:
    with st.expander("General Configurations"):
        # Configure sliders for parameters
        init_inventory = st.slider("Initial Inventory", min_value=0, max_value=100, step=1, value=5)
        reorder_point = st.slider("Reorder Point", min_value=0, max_value=100, step=1, value=10)
        order_amount = st.slider("Order Amount", min_value=0, max_value=100, step=1, value=5)
        lead_time_range = st.slider("Lead Time Range (weeks)", min_value=1, max_value=20, step=1, value=(6, 12))
        num_weeks = st.slider("Number of Weeks", min_value=1, max_value=100, step=1, value=52)

    # Configure collapsible section for cost configurations
    with st.expander("Cost Configurations"):
        # Configure sliders for cost parameters
        storage_cost_per_day = st.slider("Cost per Storage per Day (€)", min_value=1, max_value=1000, step=1, value=100)
        downtime_cost_per_day = st.slider("Cost per Downtime per Day (€)", min_value=10000, max_value=100000, step=100, value=50000)

    # Configure collapsible section for error configurations
    with st.expander("Failure Configurations"):
        # Configure sliders for error parameters
        error_scale = st.slider("Error Scale", min_value=0.01, max_value=30.0, step=0.01, value=2.0)
        error_shape = st.slider("Error Shape", min_value=0.1, max_value=10.0, step=0.1, value=4.0)
        ttf = st.slider("Time Until Failure (TTF)", min_value=1.0, max_value=30.0, step=1.0, value=2.0)

# Simulate inventory stock based on the given parameters
inventory_levels, reorder_flags = simulate_inventory_stock(
    init_inventory, reorder_point, order_amount, error_scale, error_shape, lead_time_range, num_weeks
)

# region visualization_nonopt
# Set page title
st.title("Inventory Stock Visualization")

# Calculate accumulated costs
storage_costs = sum(max(level, 0) * storage_cost_per_day for level in inventory_levels)
downtime_costs = sum(downtime_cost_per_day if level <= 0 else 0 for level in inventory_levels)
overall_costs = storage_costs + downtime_costs

# Display accumulated costs
st.markdown("## Accumulated Costs")
st.table({"Storage Costs": f"{storage_costs} €" ,"Downtime Costs": f"{downtime_costs} €",f"Overall Costs" : f"{overall_costs} €"})

# Plot inventory stock over time
plt.figure(figsize=(10, 6))
plt.plot(range(num_weeks), inventory_levels, label="Inventory Level")
plt.plot(range(num_weeks), reorder_flags, label="Reorder Flag")
plt.xlabel("Weeks")
plt.ylabel("Inventory Stock")
plt.title("Inventory Stock Over Time")
plt.legend()
plt.grid(True)

# Display plot using Streamlit
st.pyplot(plt)
# endregion

with st.expander("Weibull Error Shape Curve"):
    # Plot Weibull error shape curve
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 10, 1000)  # Timespan range
    y = weibull_min.pdf(x, error_shape, scale=error_scale)
    plt.plot(x, y)
    plt.axvline(x=ttf, color='red', linestyle='--', label='TTF')
    plt.xlabel("Timespan")
    plt.ylabel("Probability of Failure")
    plt.title("Weibull Error Shape Curve")
    plt.legend()
    plt.grid(True)

    # Display plot using Streamlit
    st.pyplot(plt)
    st.markdown("""
    The Weibull Error Shape Curve shown above represents the probability of a failure of a part after a given timespan. 
    The x-axis represents the timespan, and the y-axis represents the probability of failure. 
    The shape parameter (beta) influences the steepness of the curve, while the scale parameter (lambda) affects the position of the curve along the timespan. 
    The red dashed line indicates the Time Until Failure (TTF) point.
    By adjusting the Error Scale, Error Shape, and Time Until Failure (TTF) sliders, you can modify the Weibull distribution and observe its impact on the inventory stock over time.
    """)

with st.expander("Learning optimized Reorder Points"):
    if st.button("Run Optimization"):
        optimal_reorder_points, overall_costs = optimize(storage_cost_per_day,downtime_cost_per_day,reorder_point,init_inventory,order_amount,error_scale,error_shape,lead_time_range)

        inventory_levels, reorder_flags = simulate_optimized_inventory_stock(
            optimal_reorder_points, init_inventory, order_amount, error_scale, error_shape, lead_time_range, num_weeks
        )

        storage_costs = sum(max(level, 0) * storage_cost_per_day for level in inventory_levels)
        downtime_costs = sum(downtime_cost_per_day if level <= 0 else 0 for level in inventory_levels)
        overall_costs = storage_costs + downtime_costs

        # Display accumulated costs
        st.markdown("## Accumulated Costs")
        st.table({"Storage Costs": f"{storage_costs} €" ,"Downtime Costs": f"{downtime_costs} €",f"Overall Costs" : f"{overall_costs} €"})

        # Plot inventory stock over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_weeks), inventory_levels, label="Inventory Level")
        plt.plot(range(num_weeks), reorder_flags, label="Reorder Flag")
        plt.xlabel("Weeks")
        plt.ylabel("Inventory Stock")
        plt.title("Inventory Stock Over Time")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)
