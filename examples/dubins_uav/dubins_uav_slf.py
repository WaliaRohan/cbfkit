from jax import Array, jit
import jax.numpy as jnp

# Module for generating code for new dynamics/controllers/cbfs/clfs
from cbfkit.codegen.create_new_system import generate_model

### Dubins UAV Model

# States x, y, theta, v
x = "x[0]"
y = "x[1]"
theta = "x[2]"
v = "x[3]"

# Dictionary of parameters
params = {}

# Equations of motion: xdot = f(x) + g(x)u
f = [
    f"{v} * cos({theta})",
    f"{v} * sin({theta})",
    "0",
    "0",
]
g = [
    "[0]",
    "[0]",
    "[1]",
    "[0]",
]

## If there were parameters in the dynamics model
# params["dynamics"] = {"parameter_name: data_type": default_value}

### Nominal Controller: stabilize to y=0 at constant positive x velocity

# Y setpoint
yd = 0.0
yd_dot = f"{yd}-{y}"

# Define Lyapunov fcn
ey = f"{yd_dot} - {f[1]}"
lyap = f"1/2 * ({ey})**2"

# CLF-inspired control: Vdot = -k*V
nominal_control_law = (
    f"[(-(kv * {lyap}) - ({y}) * ({f[1]}) - ({f[1]})**2) / (1e-6 + {y} * {v} * cos({theta}))]"
)
params["controller"] = {"kv: float": 1.0}

### Candidate CBF: hdot * tau + h for h = dx**2 + dy**2 - r**2

# Define dx, dy, dvx, dvy according to states
dx = f"cx - x[0]"
dy = f"cy - x[1]"
dvx = f"-{f[0]}"
dvy = f"-{f[1]}"

# Define h, hdot
h = f"({dx})**2 + ({dy})**2 - r**2"
hdot = f"2 * ({dx}) * ({dvx}) + 2 * ({dy}) * ({dvy})"

# Specify candidate CBF and params
candidate_cbfs = [f"({hdot}) * tau + {h}"]
params["cbf"] = [{"cx: float": 1.0, "cy: float": 1.0, "r: float": 1.0, "tau: float": 1.0}]

### Generate New Code

# Directory and name for generated model
target_directory = "./src/models"
model_name = "dubins_uav"

# Format for codegen module
drift_dynamics = "[" + ",".join(f) + "]"
control_matrix = "[" + ",".join(g) + "]"

# Generate code for defined model
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=candidate_cbfs,
    nominal_controller=nominal_control_law,
    params=params,
)

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


# Import newly generated Dubins UAV code
from src.models import dubins_uav

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
