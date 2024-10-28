import os
import sys

import jax.numpy as jnp
from jax import Array, jit

# Add 'src/cbfkit' to the Python path
script_dir = os.path.dirname(os.path.realpath(__file__))
cbfkit_path = os.path.join(script_dir, '..', '..', 'src')
sys.path.append(cbfkit_path)
print(script_dir, "------", cbfkit_path)

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

# Define h, hdot
h = f"x[1] - d"
hdot = f"{f[1]}"

# Specify candidate CBF and params
# candidate_cbfs = [f"({hdot}) * tau + {h}"]
candidate_cbfs = [f"{h}"]
params["cbf"] = [{"d: float": 1.0, "tau: float": 1.0}]

### Generate New Code

# Directory and name for generated model
target_directory = "./src/models"
model_name = "dubins_uav_wall"

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
