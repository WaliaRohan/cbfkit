import os
import sys

import jax.numpy as jnp

# Add 'src/cbfkit' to the Python path
script_dir = os.path.dirname(os.path.realpath(__file__))
cbfkit_path = os.path.join(script_dir, '..', '..', 'src')
sys.path.append(cbfkit_path)
print(script_dir, "------", cbfkit_path)

# Module for generating code for new dynamics/controllers/cbfs/clfs
from cbfkit.codegen.create_new_system import generate_model

### Single integrator model

# States x
x = "x[0]"

# Dictionary of parameters
params = {}

# Equations of motion: xdot = f(x) + g(x)u
f = ["0"]
g = ["1"]

## If there were parameters in the dynamics model
# params["dynamics"] = {"parameter_name: data_type": default_value}

### Nominal Controller: stabilize to x=x_d
# Y setpoint
xd = 10.0
e_x = f"{xd}-{x}"

# Define Lyapunov fcn
lyap = f"1/2 * ({e_x})**2"

# CLF inspired control law using Arsten-Sontag's theorem: 
nominal_control_law = (
    f"[2*({e_x})]"
)
# params["controller"] = {"kv: float": 1.0}

# Define h, hdot
h = f"x[0] - d"
hdot = f"{f[0]}"

# Specify candidate CBF and params
candidate_cbfs = [f"{h}"]
params["cbf"] = [{"d: float": 1.0}]

### Generate New Code

# Directory and name for generated model
target_directory = "./src/models"

if not os.path.exists(target_directory):
    os.makedirs(target_directory)

model_name = "single_integrator_2"

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
