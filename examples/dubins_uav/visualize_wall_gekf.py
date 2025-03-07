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
from jax import jit, jacfwd

### Import existing CBFkit code for simulation
# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

# Import dubins_uav_wall barrier function package
import src.models.dubins_uav_wall.certificate_functions.barrier_functions.barrier_1 as barrier_certificate

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)

# Access to CBF-CLF-QP control law
from cbfkit.controllers.model_based.cbf_clf_controllers.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller,
)

# from cbfkit.estimators import naive as estimator
from cbfkit.estimators import ct_gekf_dtmeas


# To add stochastic perturbation to system dynamics
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

# Perfect and imperfect sensors
# from cbfkit.sensors import unbiased_gaussian_noise as noisy_sensor_mult # can just use a non-state dependent sensor here
from cbfkit.sensors import unbiased_gaussian_noise_mult as noisy_sensor_mult

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator


@jit
def sigma(x):
    # return jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE
    return jnp.zeros((4, 4))

model_name = "dubins_uav_wall"

# Import newly generated Dubins UAV code
from models import dubins_uav_wall

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"
DT = 1e-3
TF = 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 100.0, np.radians(245), 1.0])
ACTUATION_LIMITS = jnp.array([1.0])  # Box control input constraint, i.e., -1 <= u <= 1

# Dynamics function: dynamics(x) returns f(x), g(x), d(x)
dynamics = dubins_uav_wall.plant()

wall_x = 10.0
class_k_gain = 2.0
optimized_alpha = False

barriers = [
    barrier_certificate.cbf1_package(
        certificate_conditions=zeroing_barriers.linear_class_k(class_k_gain),
        d = wall_x)
]
barrier_packages = concatenate_certificates(*barriers)
###

# Instantiate nominal controller
kv = 0.01  # control gain defined in previous expression
nominal_controller = dubins_uav_wall.controllers.controller_1(kv=kv)

### Instantiate CBF-CLF-QP control law
cbf_clf_controller = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    barriers=barrier_packages,
    tunable_class_k=optimized_alpha,
    # relaxable_clf=True,
)

# Q = 0.0027 * jnp.eye(len(INITIAL_STATE))  # process noise
Q = 0.05 * jnp.eye(len(INITIAL_STATE))  # process noise
R = 7.997e-5 * jnp.eye(len(INITIAL_STATE))  # measurement noise -> not used for GEKF
plant_jacobians = jacfwd(dynamics)
dfdx = plant_jacobians
h = lambda x: x
dhdx = lambda _x: np.eye((len(INITIAL_STATE)))

estimator = ct_gekf_dtmeas(
    Q=Q,
    R=R,
    dynamics=dynamics,
    dfdx=dfdx,
    h=h,
    dhdx=dhdx,
    dt=DT,
)

x, u, estimates, p, dkeys, dvalues, measurements = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
    integrator=integrator,
    controller=cbf_clf_controller,
    sensor=noisy_sensor_mult,
    # sensor = noisy_sensor_sd(),
    estimator=estimator,
    filepath=SAVE_FILE,
)

print("State: ", x[0])
print("Measurement: ", measurements[0])

###################################################################################################
## Visualization ##

import matplotlib.animation as animation
import matplotlib.pyplot as plt

total_time = DT * len(x)

print("Total time: ", total_time)

fig, ax = plt.subplots()
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title(f"System Trajectory (T = {total_time:.2f} s)")

save = True
animate = False
plot_heading = False

save_directory = "plots/" + model_name + "_gekf/"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print()

if save:
    ax.axvline(x=wall_x, color='purple', linestyle=':', linewidth=2, label=f"obstacle")
    measurements = np.array(measurements)
    estimates = np.array(estimates)

    if len(x) == len(measurements):
        ax.plot(measurements[:, 0], measurements[:, 1],  color='green', label='Measured Trajectory', linewidth=0.5)
        
    if len(x) == len(estimates):
        ax.plot(estimates[:, 0], estimates[:, 1], color='orange', label='Estimated Trajectory', linewidth=0.5)

    ax.plot(x[:, 0], x[:, 1], color='blue', label='True Trajectory')
    ax.legend()

    if(plot_heading):
        # Plot direction arrows
        arrow_scale = 0.5  # Scaling factor for arrow length
        arrow_spacing = max(1, len(x) // 20)  # Plot fewer arrows for clarity

        for i in range(0, len(x), arrow_spacing):  # Adjusting arrow density
            dx = np.cos(x[i, 2]) * x[i, 3] * arrow_scale  # Scaled velocity projection along x-axis
            dy = np.sin(x[i, 2]) * x[i, 3] * arrow_scale  # Scaled velocity projection along y-axis
            ax.arrow(
                x[i, 0], x[i, 1], dx, dy, 
                head_width=0.5, head_length=0.15, fc='green', ec='green', alpha=0.8
            )

    fig.savefig(save_directory + model_name + " system_trajectory" + ".png")

    # Plot CBF values
    fig2, ax2 = plt.subplots()
    time_steps = np.linspace(0, total_time, len(x))
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

    fig5, ax5 = plt.subplots()
    ax5.plot(time_steps, measurements[:, 0], label='Measured X', linewidth=0.5)
    ax5.plot(time_steps, estimates[:, 0], label='Estimated X', linestyle='--', linewidth=0.7)
    ax5.plot(time_steps, x[:, 0], label='True X')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("X")
    ax5.legend()
    fig5.savefig(save_directory + model_name + " true_vs_measured_x" + ".png")

    fig6, ax6 = plt.subplots()
    ax6.plot(time_steps, measurements[:, 1], color='green', label='Measured Y', linewidth=0.5)
    ax6.plot(time_steps, estimates[:, 1], color='orange', label='Estimated Y', linestyle='--', linewidth=0.7)
    ax6.plot(time_steps, x[:, 1], color='blue', label='True Y')
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Y")
    ax6.legend()
    fig6.savefig(save_directory + model_name + " true_vs_measured_Y" + ".png")
        

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


def compute_rmse(measurements, x):
    """
    Compute the root mean squared error (RMSE) between measurements (p_hat) and x (p)
    for the first and second states.

    Args:
        measurements (np.ndarray): N-dimensional 1D numpy array of estimated values.
        x (np.ndarray): N-dimensional 1D numpy array of true values.

    Returns:
        tuple: RMSE for the first state and RMSE for the second state.
    """
    errors = measurements - x
    rmse_first = np.sqrt(np.mean(errors[:, 0]**2))  # First state
    rmse_second = np.sqrt(np.mean(errors[:, 1]**2))  # Second state
    
    return rmse_first, rmse_second


rmse_1, rmse_2 = compute_rmse(measurements, x)
# print(f"RMSE x: {rmse_1}")
print(f"GEKF RMSE y: {rmse_2}")