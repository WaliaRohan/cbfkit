import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the system path
sys.path.append(os.path.join(current_dir, "src"))

from jax import jit
import jax.numpy as jnp

### Import existing CBFkit code for simulation
# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim

# Access to CBF-CLF-QP control law
from cbfkit.controllers.model_based.cbf_clf_controllers.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller,
)

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.rectify_relative_degree import (
    rectify_relative_degree,
)

# Assuming perfect, complete state information
from cbfkit.sensors import perfect as sensor

# With perfect sensing, we can use a naive estimate of the state
from cbfkit.estimators import naive as estimator

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator

# To add stochastic perturbation to system dynamics
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation


@jit
def sigma(x):
    # return jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE
    return jnp.zeros((4, 4))


model_name = "dubins_uav"

# Import newly generated Dubins UAV code
from models import dubins_uav

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"
DT = 1e-2
TF = 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([0.0, 0.0, 0.0, 1.0])
ACTUATION_LIMITS = jnp.array([1.0])  # Box control input constraint, i.e., -1 <= u <= 1

# Dynamics function: dynamics(x) returns f(x), g(x), d(x)
dynamics = dubins_uav.plant()

# Specific instantiation of barrier functions
centers_x = [2.5, 5.0]
centers_y = [0.25, -0.25]
radii = [0.5, 0.5]
taus = [0.5, 0.5]

### This code accomplishes the following:
# - passes the parameters cx, cy, r, tau to the generic (unspecified) candidate CBF to create a specific one
# - passes the instantiated cbf through the rectify_relative_degree func, which returns a (possibly new) CBF
#       that is guaranteed to have relative-degree one with respect to the system_dynamics and is constructed
#       using exponential CBF principles
# - specifies the type of CBF condition to enforce (in this case a zeroing CBF condition with a linear class K func)
# - then packages the (two, in this case) barrier functions into one object for the controller to use
barriers = [
    rectify_relative_degree(
        function=dubins_uav.certificate_functions.barrier_functions.barrier_1.cbf(
            cx=cx,
            cy=cy,
            r=r,
            tau=tau,
        ),
        system_dynamics=dynamics,
        state_dim=len(INITIAL_STATE),
        form="exponential",
        roots=jnp.array([-1.0, -1.0, -1.0]),
    )(certificate_conditions=zeroing_barriers.linear_class_k(2.0), cx=cx, cy=cy, r=r, tau=tau)
    for cx, cy, r, tau in zip(centers_x, centers_y, radii, taus)
]
barrier_packages = concatenate_certificates(*barriers)
###

# Change this to True if you want the linear gain in the CBF condition's class K function
# to be a decision variable in the optimization problem
optimized_alpha = True

# Instantiate nominal controller
kv = 2.0  # control gain defined in previous expression
nominal_controller = dubins_uav.controllers.controller_1(kv=kv)

### Instantiate CBF-CLF-QP control law
cbf_clf_controller = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    barriers=barrier_packages,
    tunable_class_k=optimized_alpha,
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

x, u, z, p, dkeys, dvalues = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
    integrator=integrator,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
)

###################################################################################################
## Visualization ##

import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Visualization ##

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

save = True
animate = False

if save:
    ax.plot(x[:, 0], x[:, 1])
    for cx, cy, r in zip(centers_x, centers_y, radii):
        ax.add_patch(
            plt.Circle(
                jnp.array([cx, cy]),
                r,
                color="k",
                fill=True,
                linewidth=1,
            )
        )

    # plt.show()
    plt.savefig("output.png")

if animate:
    (line,) = ax.plot([], [], lw=5)

    for cx, cy, r in zip(centers_x, centers_y, radii):
        ax.add_patch(
            plt.Circle(
                jnp.array([cx, cy]),
                r,
                color="k",
                fill=True,
                linewidth=1,
            )
        )

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
