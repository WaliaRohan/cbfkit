'''
This file is for reading the simulation data (usually tutorials/model_name.simulation_data.csv)
for any simulation.


Currently, it does not read barrier function (bfs) values.
'''

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # For pandas

import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the directory you want to attach (relative to current_dir)
sub_dir = "tutorials/dubins_uav_wall/simulation_data.csv"

# Create the full path
file_path = os.path.join(current_dir, sub_dir)

# # Read the CSV file
df = pd.read_csv(file_path)

data = df.to_dict(orient='list')

states = []
estimates = []
measurements = []

for x, z, y in zip(data['x'], data['z'], data['y']):
    
    x_array = np.squeeze(np.array([eval(re.sub(r'\s+', ',', x.strip()))]))
    print(x_array)
    states.append(x_array)

    z = re.sub(r'\[\s+', '[', z)
    z_cleaned = re.sub(r'\bnan\b', 'np.nan', z)
    z_array = np.squeeze(np.array([eval(re.sub(r'\s+', ',', z_cleaned.strip()))]))
    print(z_array)
    estimates.append(z_array)

    y = re.sub(r'\[\s+', '[', y)
    y_cleaned = re.sub(r'\bnan\b', 'np.nan', y)
    y_array = np.squeeze(np.array([eval(re.sub(r'\s+', ',', y_cleaned.strip()))]))
    print(y_array)
    measurements.append(y_array)

u_values = []
u_nom_values = []
bfs_values = []

for u, u_nom, sub_data_str in zip(data['u'], data['u_nom'], data['sub_data']):
    # Clean and convert u values
    u_cleaned = re.sub(r'\[\s+', '[', u)
    u_cleaned = re.sub(r'\bnan\b', 'np.nan', u_cleaned)  # Replace 'nan' with np.nan
    u_array = np.squeeze(np.array([eval(re.sub(r'\s+', ',', u_cleaned.strip()))]))
    print(u_array)
    u_values.append(u_array)

    # Clean and convert u_nom values
    u_nom_cleaned = re.sub(r'\[\s+', '[', u_nom)
    u_nom_cleaned = re.sub(r'\bnan\b', 'np.nan', u_nom_cleaned)
    u_nom_array = np.squeeze(np.array([eval(re.sub(r'\s+', ',', u_nom_cleaned.strip()))]))
    print(u_nom_array)
    u_nom_values.append(u_nom_array)

    print(sub_data_str)
    # # Extract and convert bfs values from sub_data
    # bfs_array = sub_data['bfs'].astype(float)  # Extract 'bfs' value and convert to float
    # print(bfs_array)
    # bfs_values.append(bfs_array)

# Convert the lists to NumPy arrays
u_values_array = np.array(u_values)
u_nom_values_array = np.array(u_nom_values)
bfs_values_array = np.array(bfs_values)


# Extracting x, y, theta for states, estimates, and measurements
state_x = [state[0] for state in states]  # x values from states
state_y = [state[1] for state in states]  # y values from states
state_theta = [state[2] for state in states]  # theta values from states

estimate_x = [estimate[0] for estimate in estimates]  # x values from estimates
estimate_y = [estimate[1] for estimate in estimates]  # y values from estimates
estimate_theta = [estimate[2] for estimate in estimates]  # theta values from estimates

measurement_x = [measurement[0] for measurement in measurements]  # x values from measurements
measurement_y = [measurement[1] for measurement in measurements]  # y values from measurements
measurement_theta = [measurement[2] for measurement in measurements]  # theta values from measurements

# Create 3 plots
fig, axes = plt.subplots(4, 1, figsize=(10, 15))

wall_x = 1.0

# Trajectory plot for states
axes[0].plot(state_x, state_y, label="States", color='b')
axes[0].axvline(x=wall_x, color='k', linestyle=':', linewidth=2, label=f"x = {wall_x}")
axes[0].set_title("Trajectory of States (x, y)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True)
axes[0].legend()

# Trajectory plot for estimates
axes[1].plot(estimate_x, estimate_y, label="Estimates", color='r')
axes[1].axvline(x=wall_x, color='k', linestyle=':', linewidth=2, label=f"x = {wall_x}")
axes[1].set_title("Trajectory of Estimates (x, y)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].grid(True)
axes[1].legend()

# Trajectory plot for measurements
axes[2].plot(measurement_x, measurement_y, label="Measurements", color='g')
axes[2].axvline(x=wall_x, color='k', linestyle=':', linewidth=2, label=f"x = {wall_x}")
axes[2].set_title("Trajectory of Measurements (x, y)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].grid(True)
axes[2].legend()

# Plot for u and u_nom
axes[3].plot(u_values_array, label="u", color='b', linestyle='-', linewidth=2)
axes[3].plot(u_nom_values_array, label="u_nom", color='r', linestyle='--', linewidth=2)
axes[3].set_title("Plot of u and u_nom")
axes[3].set_xlabel("Index")
axes[3].set_ylabel("Value")
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()

# Save the plots as PNG files
plt.savefig("states_trajectory.png")
plt.savefig("estimates_trajectory.png")
plt.savefig("measurements_trajectory.png")
plt.savefig("u_and_u_nom_plot.png")

plt.show()

