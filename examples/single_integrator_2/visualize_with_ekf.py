import os
import sys

import numpy as np

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add 'src/cbfkit' to the Python path
cbfkit_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(cbfkit_path)
print(current_dir, "------", cbfkit_path)

# Add the parent directory to the system path
sys.path.append(os.path.join(current_dir, "src"))

import jax.numpy as jnp
import jax.random as random

# Import single_integrator_multip_noise barrier function package
import src.models.single_integrator_multip_noise.certificate_functions.barrier_functions.barrier_1 as barrier_certificate
from jax import jit, jacfwd

### Import existing CBFkit code for simulation
# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)

# Access to CBF-CLF-QP control law
from cbfkit.controllers.model_based.cbf_clf_controllers.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller,
)

# Perfect state estimation
# from cbfkit.estimators import naive as estimator
from cbfkit.estimators import ct_ekf_dtmeas

# To add stochastic perturbation to system dynamics
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

# Perfect and imperfect sensors
from cbfkit.sensors import perfect
from cbfkit.sensors import unbiased_gaussian_noise_sd as noisy_sensor

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator


@jit
def sigma(x):
    # return jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE
    return jnp.zeros((1, 1))


model_name = "single_integrator_multip_noise"

# Import newly generated code
from models import single_integrator_multip_noise

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"
DT = 1e-3 # use 10^-3 or 10^-4
TF = 20.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0])
ACTUATION_LIMITS = jnp.array([1.0])  # Box control input constraint, i.e., -1 <= u <= 1

# Dynamics function: dynamics(x) returns f(x), g(x), d(x)
dynamics = single_integrator_multip_noise.plant()

wall_x = 3.0
class_k_gain = 0.2
key_seed = 0
base_key = random.PRNGKey(key_seed)  # Starting key
keys = random.split(base_key, 1000)  # Generate unique keys
key = keys[109]
print("Using key: ", key)

barriers = [
    barrier_certificate.cbf1_package(
        certificate_conditions=zeroing_barriers.linear_class_k(class_k_gain),
        d = wall_x)
]
barrier_packages = concatenate_certificates(*barriers)
###

# Change this to True if you want the linear gain in the CBF condition's class K function
# to be a decision variable in the optimization problem
optimized_alpha = False

# Instantiate nominal controller
nominal_controller = single_integrator_multip_noise.controllers.controller_1()

### Instantiate CBF-CLF-QP control law
cbf_clf_controller = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    barriers=barrier_packages,
    tunable_class_k=optimized_alpha,
)

Q = 0.05 * jnp.eye(len(INITIAL_STATE))  # process noise
R = 0.05 * jnp.eye(len(INITIAL_STATE))  # measurement noise
plant_jacobians = jacfwd(dynamics)
# plant_jacobians = lambda : 1
dfdx = plant_jacobians
h = lambda x: x
dhdx = lambda _x: np.eye((len(setup.initial_state)))

estimator = ct_ekf_dtmeas(
    Q=Q,
    R=R,
    dynamics=dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=DT,
)

### Simulate the system
# - beginning at initial state x0
# - with timestep dt
# - for a fixed number of timesteps num_steps
# - with system model given by dynamics
# - and controlled by controller
# - with state information sensed by sensor
# - and a state estimate computed by estimator
# - perturbed by perturbation (on plant, not measurement)
# - with numerical integration scheme specified by integrator
# - and data saved out to filepath

# Returns:
# x: states
# u: controls
# z: estimates
# p: covariances
# dkeys: data_keys
# dvalues: data_values
# measurements: sensor values
# (dkeys, dvalues) are returned by cbf_clf_qp_generator.jittable_controller in:
#       cbfkit.controllers.model_based.cbf_clf_controllers.cbf_clf_qp_generator
# Usually this data is returned by the controller in:
#                           cbfkit.simulation.simulator.stepper()


x, u, z, p, dkeys, dvalues, measurements = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
    integrator=integrator,
    controller=cbf_clf_controller,
    sensor=noisy_sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
    key=key
)

###################################################################################################
## Analysis

x = np.array(x) if not isinstance(x, np.ndarray) else x
measurements = np.array(measurements)

actual_violations = x[x > wall_x] 
total_actual_violations = jnp.sum(x > wall_x).item() # Count how many values in x are greater than wall_x
actual_violation_ratio = total_actual_violations / len(x) if len(x) > 0 else 0  # Avoid division by zero

measured_violations = measurements[measurements > wall_x]
total_measured_violations = jnp.sum(measurements > wall_x).item()  # Count how many values in x are greater than wall_x
measured_violation_ratio = total_measured_violations / len(x) if len(x) > 0 else 0  # Avoid division by zero

print("Total Actual Violations:", total_actual_violations)
print("Actual Violation Ratio:", actual_violation_ratio)
print("Total Measured Violations:", total_measured_violations)
print("Measured Violation Ratio:", measured_violation_ratio)

###################################################################################################
## Visualization ##

import matplotlib.animation as animation
import matplotlib.pyplot as plt

## Visualization ##

total_time = DT * len(x)

print("Total time: ", total_time)

fig, ax = plt.subplots()
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title(f"System Trajectory (T = {total_time:.2f} s)")

# Define start and goal coordinates
x_d = 10.0
start = (INITIAL_STATE[0], 0)
goal = (x_d, 0)

# Plotting
ax.plot(start[0], start[1], 'go', label='Start')  # Start as a green circle
ax.plot(goal[0], goal[1], 'ro', label='Goal')  # Goal as a red circle

# Plot a vertical line at x = wall_x
ax.axvline(x=wall_x, color='black', linestyle='--')

save = True
animate = False

save_directory = "plots/" + model_name + "/"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

text_str = f"Final true X: {x[-1]}\nFinal measured x: {measurements[-1]}"


if save:

    y = np.zeros_like(x)

    time_steps = np.linspace(0, total_time, len(x))
    measurements = np.array(measurements) if not isinstance(measurements, np.ndarray) else measurements 

    if len(x) == len(measurements):
        # Plot measurements
        ax.plot(measurements, y, label='Measured Trajectory', linewidth=0.5)
        # Optionally add a legend to differentiate the data
        ax.legend()
        # ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
        # verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

    ax.plot(x, y, label='True Trajectory')
    
    fig.savefig(save_directory + model_name + " system_trajectory" + ".png")

    # Plot CBF values
    fig2, ax2 = plt.subplots()
    
    # Extract values of 'bfs' key from the dictionaries at index 3 in each sublist
    bfs_values = [
        data_dict["bfs"] for sublist in dvalues if "bfs" in sublist[3] for data_dict in [sublist[3]]
    ]

    if len(bfs_values) > 0:
        ax2.plot(time_steps, bfs_values)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("CBF Values")
        ax2.set_title("CBF Values")
        fig2.savefig(save_directory + model_name + " barrier_function_values" + ".png")

    # Plot nominal and actual control effort
    fig3, ax3 = plt.subplots()
    u = [sublist[4] for sublist in dvalues]
    u_nom = [sublist[5][0] for sublist in dvalues]
    ax3.plot(time_steps, u_nom, marker=".", linestyle="--", color="r", label="u_nom", markersize=6)
    ax3.plot(time_steps, u, marker=".", linestyle="-", color="b", label="u", markersize=1)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Control Input Values")
    ax3.set_title("Control Input")
    ax3.legend()
    fig3.savefig(save_directory + model_name + " control_values" + ".png")

    # Plot difference in control efforts
    difference = [ui_nom - ui for ui_nom, ui in zip(u_nom, u)]

    fig4, ax4 = plt.subplots()
    ax4.plot(
        time_steps,
        difference,
        marker="o",
        linestyle="-",
        color="g",
        label="u_nom - u",
        markersize=1,
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Difference")
    ax4.set_title("Difference between u_nom and u")
    ax4.legend()

    # Save the plots
    fig4.savefig(save_directory + model_name + " control_values_diff" + ".png")

    fig5, ax5 = plt.subplots(layout="constrained")
    ax5.plot(time_steps, x[:, 0], label='True X')
    ax5.plot(time_steps, measurements[:, 0], label='Measured X', linewidth=0.5)
    ax5.legend()
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("X (m)")
    ax5.set_title("True vs Measured X")
    # plt.gcf().text(0.95, 1.02, text_str, fontsize=10, verticalalignment='top', 
    #            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))
    fig5.savefig(save_directory + model_name + " true_vs_measured_x" + ".png")
    
    print(x[-1], measurements[-1])

if animate:
    (line,) = ax.plot([], [], lw=5)

    def animate(frame):
        line.set_data(x[:frame, 0], x[:frame, 1])
        return (line,)

    # Create the animation - not using "init" function here as it does not serve
    # any purpose. The obstacles ("plt.Circle" objects) in the animation are set
    # globally - I wasn't able to set them using the init function (I believe
    # they need to be converted to a "ax.plot" object by generating x-y data for
    # each circle)
    ani = animation.FuncAnimation(
        fig, animate, init_func=None, frames=len(x), interval=20, blit=True
    )

    if animation.writers.is_available("ffmpeg"):
        ani.save("./output.mp4", writer="ffmpeg", fps=30)
    elif animation.writers.is_available("imagemagick"):
        ani.save("./output.mp4", writer="imagemagick", fps=30)
    elif animation.writers.is_available("pillow"):
        ani.save("./output.gif", writer="pillow", fps=30)
    else:
        raise Exception(
            "Following writers are not available: ffmpeg, imagemagick, pillow. Can't save animation!"
        )
