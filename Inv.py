#!/usr/bin/env python
# coding: utf-8

# In[67]:


import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, value
import plotly.express as px

# Step 1: User Input Section
st.title("Decision Support System for Manufacturing Optimization")
st.sidebar.header("Manufacturing Approach")

# Dropdown for manufacturing approach selection
approach = st.sidebar.selectbox(
    "Select the Manufacturing Approach",
    ("Fully Additive Manufacturing", "Fully Traditional Manufacturing", "Hybrid Manufacturing"),
)

# Input parameters based on the approach
if approach == "Fully Additive Manufacturing":
    st.sidebar.subheader("Additive Manufacturing Inputs")
    num_printers = st.sidebar.number_input("Number of 3D Printers", min_value=1, value=3, step=1)
    printer_capacity = st.sidebar.number_input("Printer Capacity (units/day)", min_value=1, value=50, step=1)
    setup_cost = st.sidebar.number_input("Setup Cost per Batch ($)", min_value=1, value=200, step=1)
    production_cost = st.sidebar.number_input("Production Cost per Unit ($)", min_value=1, value=25, step=1)
    raw_material_cost = st.sidebar.number_input("Raw Material Cost ($/kg)", min_value=1, value=10, step=1)
    raw_material_holding_cost = st.sidebar.number_input("Raw Material Holding Cost ($/unit/day)", min_value=0.1, value=0.2, step=0.1)
    ordering_cost = 0  # Not applicable to additive manufacturing
    holding_cost = raw_material_holding_cost  # Holding cost for raw materials
    purchase_cost = 0  # Not applicable to additive manufacturing
    num_machines = 0  # Not applicable to additive manufacturing
    machine_capacity = 0  # Not applicable to additive manufacturing


elif approach == "Fully Traditional Manufacturing":
    st.sidebar.subheader("Traditional Manufacturing Inputs")
    num_machines = st.sidebar.number_input("Number of Machines", min_value=1, value=5, step=1)
    machine_capacity = st.sidebar.number_input("Machine Capacity (units/day)", min_value=1, value=80, step=1)
    ordering_cost = st.sidebar.number_input("Ordering Cost per Batch ($)", min_value=1, value=100, step=1)
    holding_cost = st.sidebar.number_input("Holding Cost per Unit/Day ($)", min_value=0.1, value=0.5, step=0.1)
    purchase_cost = st.sidebar.number_input("Purchase Cost per Unit ($)", min_value=1, value=20, step=1)
    setup_cost = 0  # Not applicable to traditional manufacturing
    production_cost = 0  # Not applicable to traditional manufacturing

elif approach == "Hybrid Manufacturing":
    st.sidebar.subheader("Additive Manufacturing Inputs")
    num_printers = st.sidebar.number_input("Number of 3D Printers", min_value=1, value=3, step=1)
    printer_capacity = st.sidebar.number_input("Printer Capacity (units/day)", min_value=1, value=50, step=1)
    setup_cost = st.sidebar.number_input("Setup Cost per Batch ($)", min_value=1, value=200, step=1)
    production_cost = st.sidebar.number_input("Production Cost per Unit ($)", min_value=1, value=25, step=1)
    raw_material_holding_cost = st.sidebar.number_input("Raw Material Holding Cost ($/unit/day)", min_value=0.1, value=0.2, step=0.1)

    st.sidebar.subheader("Traditional Manufacturing Inputs")
    num_machines = st.sidebar.number_input("Number of Machines", min_value=1, value=5, step=1)
    machine_capacity = st.sidebar.number_input("Machine Capacity (units/day)", min_value=1, value=80, step=1)
    ordering_cost = st.sidebar.number_input("Ordering Cost per Batch ($)", min_value=1, value=100, step=1)
    holding_cost = st.sidebar.number_input("Holding Cost per Unit/Day ($)", min_value=0.1, value=0.5, step=0.1)
    purchase_cost = st.sidebar.number_input("Purchase Cost per Unit ($)", min_value=1, value=20, step=1)


# Common Inputs
st.sidebar.subheader("Demand and Lead Times")
daily_demand_mean = st.sidebar.number_input("Mean Daily Demand (units)", min_value=1, value=200, step=1)
daily_demand_std = st.sidebar.number_input("Demand Standard Deviation (units)", min_value=1, value=50, step=1)
lead_time_mean = st.sidebar.number_input("Mean Lead Time (days)", min_value=1, value=7, step=1)
lead_time_std = st.sidebar.number_input("Lead Time Standard Deviation (days)", min_value=1, value=2, step=1)
service_level = st.sidebar.slider("Desired Service Level (%)", min_value=90, max_value=99, value=95, step=1)


# In[63]:


# Step 2: Perform Calculations
st.header("Calculated Outputs")

# Safety stock calculation
Z = {90: 1.28, 95: 1.65, 99: 2.33}[service_level]
safety_stock = Z * np.sqrt(lead_time_mean) * daily_demand_std
st.write(f"Calculated Safety Stock: {safety_stock:.2f} units")

# Capacity calculations
if approach == "Fully Additive Manufacturing":
    total_capacity = num_printers * printer_capacity
    st.write(f"Total 3D Printing Capacity: {total_capacity} units/day")
elif approach == "Fully Traditional Manufacturing":
    total_capacity = num_machines * machine_capacity
    st.write(f"Total Traditional Manufacturing Capacity: {total_capacity} units/day")
elif approach == "Hybrid Manufacturing":
    capacity_3dp = num_printers * printer_capacity
    capacity_traditional = num_machines * machine_capacity
    total_capacity = capacity_3dp + capacity_traditional
    st.write(f"Total 3D Printing Capacity: {capacity_3dp} units/day")
    st.write(f"Total Traditional Manufacturing Capacity: {capacity_traditional} units/day")
    st.write(f"Combined Capacity: {total_capacity} units/day")


# In[69]:


# Step 3: Optimization
st.header("Optimization Results")

# MILP setup
prob = LpProblem("ManufacturingOptimization", LpMinimize)

# Decision Variables
D_traditional = LpVariable("D_traditional", lowBound=0, cat="Continuous")
D_3DP = LpVariable("D_3DP", lowBound=0, cat="Continuous")

# Cost per unit calculations
setup_cost_per_unit = setup_cost / (num_printers * printer_capacity) if num_printers > 0 and printer_capacity > 0 else 0
ordering_cost_per_unit = ordering_cost / daily_demand_mean if daily_demand_mean > 0 else 0

# Objective Function: Total cost combining both methods
TC_traditional = (
    ordering_cost_per_unit * D_traditional
    + holding_cost * D_traditional
    + purchase_cost * D_traditional
)

TC_3DP = (
    setup_cost_per_unit * D_3DP
    + production_cost * D_3DP
    + raw_material_holding_cost * D_3DP
)

# Total Cost
prob += TC_traditional + TC_3DP

# Constraints
prob += D_traditional + D_3DP == daily_demand_mean, "DemandFulfillment"

# Add capacity constraints for each method
if approach in ["Fully Traditional Manufacturing", "Hybrid Manufacturing"]:
    prob += D_traditional <= num_machines * machine_capacity, "TraditionalCapacity"

if approach in ["Fully Additive Manufacturing", "Hybrid Manufacturing"]:
    prob += D_3DP <= num_printers * printer_capacity, "3DPrintingCapacity"

# Solve the problem
prob.solve()

# Retrieve optimized decision variables
D_traditional_opt = D_traditional.varValue if D_traditional.varValue is not None else 0
D_3DP_opt = D_3DP.varValue if D_3DP.varValue is not None else 0

# Calculate costs
traditional_cost = (
    ordering_cost_per_unit * D_traditional_opt
    + holding_cost * D_traditional_opt
    + purchase_cost * D_traditional_opt
)

additive_cost = (
    setup_cost_per_unit * D_3DP_opt
    + production_cost * D_3DP_opt
    + raw_material_holding_cost * D_3DP_opt
)

# Additional metrics
unmet_demand = max(0, daily_demand_mean - (D_traditional_opt + D_3DP_opt))
fill_rate = 1 - (unmet_demand / daily_demand_mean)


# In[73]:


# Display optimization results
st.write("### Cost Breakdown and Demand Allocation")
cost_data = {
    "Category": ["Traditional Manufacturing", "Additive Manufacturing"],
    "Optimal Demand (units)": [D_traditional_opt, D_3DP_opt],
    "Cost ($)": [traditional_cost, additive_cost],
}
cost_table = pd.DataFrame(cost_data)
st.table(cost_table)

# Additional Metrics
st.write("### Additional Metrics")
st.write(f"Unmet Demand: {unmet_demand:.2f} units")
st.write(f"Fill Rate: {fill_rate * 100:.2f}%")

# Visualize demand allocation
demand_data = {
    "Manufacturing Method": ["Traditional", "3D Printing"],
    "Optimal Demand (units)": [D_traditional_opt, D_3DP_opt],
}
fig_demand = px.bar(
    demand_data,
    x="Manufacturing Method",
    y="Optimal Demand (units)",
    title="Optimized Demand Allocation",
    text="Optimal Demand (units)",
)
st.plotly_chart(fig_demand)

