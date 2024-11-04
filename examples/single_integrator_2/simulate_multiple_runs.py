import os
import sys

import numpy as np

import time

import pickle

# Suppress specific DeprecationWarnings (for pandas)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd # For saving simulation data

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

# Import single_integrator_2 barrier function package
import src.models.single_integrator_2.certificate_functions.barrier_functions.barrier_1 as barrier_certificate
from jax import jit

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
from cbfkit.estimators import naive as estimator

# To add stochastic perturbation to system dynamics
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation

# Perfect and imperfect sensors
from cbfkit.sensors import unbiased_gaussian_noise_sd as noisy_sensor

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator

@jit
def sigma(x):
    # return jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE
    return jnp.zeros((1, 1))


def simulate(key, initial_state, wall_x, DT, TF, class_k_gain, filepath):

    model_name = "single_integrator_2"

    # Import newly generated Dubins UAV code
    from models import single_integrator_2

    # Simulation Parameters
    # SAVE_FILE = f"tutorials/{model_name}/simulation_data"
    N_STEPS = int(TF / DT) + 1
    INITIAL_STATE = initial_state
    ACTUATION_LIMITS = jnp.array([1.0])  # Box control input constraint, i.e., -1 <= u <= 1

    # Dynamics function: dynamics(x) returns f(x), g(x), d(x)
    dynamics = single_integrator_2.plant()

    wall_x = wall_x

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
    nominal_controller = single_integrator_2.controllers.controller_1()

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
    # measurements: sensor values
    # (dkeys, dvalues) are returned by cbf_clf_qp_generator.jittable_controller in:
    #       cbfkit.controllers.model_based.cbf_clf_controllers.cbf_clf_qp_generator
    # Usually this data is returned by the controller in:
    #                           cbfkit.simulation.simulator.stepper()

    return sim.execute(
        x0=INITIAL_STATE,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=dynamics,
        perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
        integrator=integrator,
        controller=cbf_clf_controller,
        sensor=noisy_sensor,
        estimator=estimator,
        filepath=filepath,
        key=key
    )

if __name__=="__main__":
    
    DT = 1e-3 # use 10^-3 or 10^-4
    TF = 20.0
    initial_state = jnp.array([0.0])
    wall_x = 9.0
    class_k_gain = 0.2

    runs = 1000
    start_run = 1
    end_run = 10

    base_key_seed = 0
    base_key = random.PRNGKey(base_key_seed)  # Starting key
    keys = random.split(base_key, runs)  # Generate unique keys

    results = []

    actual_violation_count = []
    measured_violation_count = []

    actual_violation_ratios = []
    measured_violation_ratios = []

    max_actual_violation = -1
    max_measured_violation = -1

    sim_data_path = os.path.join(current_dir, 'data')

    if not os.path.exists(sim_data_path):
        os.makedirs(sim_data_path)

    run = start_run

    for run in range (start_run, end_run+1):

        key = keys[run-1]
        print(f"Run {run} in range [{start_run}, {end_run}]")
        print(f"Using key {key}")

        sim_start = time.time()
        x, u, z, p, dkeys, dvalues, measurements = simulate(key=key,
                                                            initial_state=initial_state,
                                                            wall_x=wall_x,
                                                            DT=DT, TF=TF,
                                                            class_k_gain=class_k_gain,
                                                            filepath=None)
        sim_end = time.time() - sim_start

        print("Sim wall-clock time: ", sim_end)

        process_start = time.time()
        
        u_nom = [sublist[5][0] for sublist in dvalues] 
        bfs_values = [
            data_dict["bfs"] for sublist in dvalues if "bfs" in sublist[3] for data_dict in [sublist[3]]] 

        # Convert to numpy
        time_steps = np.arange(0, TF + DT, DT).reshape(-1, 1)
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        u = np.array(u) if not isinstance(u, np.ndarray) else u
        z = np.array(z) if not isinstance(z, np.ndarray) else z
        p = np.array(p) if not isinstance(p, np.ndarray) else p
        measurements = np.array(measurements) if not isinstance(measurements, np.ndarray) else measurements 
        u_nom = np.array(u_nom) if not isinstance(u_nom, np.ndarray) else u_nom
        u_nom = u_nom.reshape(-1, 1)
        bfs_values = np.array(bfs_values) if not isinstance(bfs_values, np.ndarray) else bfs_values
        
        actual_run_violations = x[x > wall_x] 
        total_actual_run_violations = jnp.sum(x > wall_x).item() # Count how many values in x are greater than wall_x
        actual_violation_count.append(total_actual_run_violations)  # Append the count to the violations list
        actual_run_violation_ratio = total_actual_run_violations / len(x) if len(x) > 0 else 0  # Avoid division by zero
        actual_violation_ratios.append(actual_run_violation_ratio)  # Append the ratio to the ratios list

        measured_run_violations = measurements[measurements > wall_x]
        total_measured_run_violations = jnp.sum(measurements > wall_x).item()  # Count how many values in x are greater than wall_x
        measured_violation_count.append(total_measured_run_violations)  # Append the count to the violations list
        measured_run_violation_ratio = total_measured_run_violations / len(x) if len(x) > 0 else 0  # Avoid division by zero
        measured_violation_ratios.append(measured_run_violation_ratio)  # Append the ratio to the ratios list

        if len(actual_run_violations) > 0 and max(actual_run_violations) > max_actual_violation:
            max_actual_violation = max(actual_run_violations)

        if len(measured_run_violations) > 0 and max(measured_run_violations) > max_measured_violation:
            max_measured_violation = max(measured_run_violations)

        results = np.hstack((time_steps, x, u, z, measurements, u_nom, bfs_values))

        column_headers = ['Time Step', 'x', 'u', 'z', 'Measurements', 'u_nom', 'bfs_values']

        # Create a DataFrame from results
        df = pd.DataFrame(results, columns=column_headers)

        process_end = time.time() - process_start

        print("Processing wall_clock time: ", process_end)

        write_start = time.time()

        # Save to CSV
        df.to_csv(sim_data_path + f'/single_integrator_simulation_results_run_{run}.csv', index=False)

        write_end = time.time() - write_start

        print("Writing wall clock time: ", write_end)

        run += 1

        # print(dkeys)

    # Calculate the average accross all runs
    actual_violation_count_mean = np.mean(actual_violation_count) if actual_violation_count else 0
    measured_violation_count_mean = np.mean(measured_violation_count) if measured_violation_count else 0
    actual_violations_average_ratio = np.mean(actual_violation_ratios) if actual_violation_ratios else 0
    measured_violations_average_ratio = np.mean(measured_violation_ratios) if measured_violation_ratios else 0

    # Print results
    print(f"Results for runs {start_run} to {end_run}")
    print("Actual Violation Count Mean:", actual_violation_count_mean)
    print("Measured Violation Count Mean:", measured_violation_count_mean)
    print("Actual Violations Average Ratio:", actual_violations_average_ratio)
    print("Measured Violations Average Ratio:", measured_violations_average_ratio)
    print("max_actual_violation:", max_actual_violation)
    print("max_measured_violation:", max_measured_violation)

    # Sim Params:
    print("Final time (TF):", TF)
    print("Step size (DT):", DT)
    print("Initial State:", initial_state)
    print("Wall x:", wall_x)
    print("Base Key Seed:", base_key_seed)
    
    # Save results for these runs: 
    # Define the filename
    filename = f'Single_integrator_multiple_runs_{start_run}_{end_run}.pkl'

    # Construct the full file path
    file_path = os.path.join(sim_data_path, filename)

    # Create a dictionary to store the variables
    data_to_save = {
        "start_run": start_run,
        "end_run": end_run,
        "actual_violation_count_mean": actual_violation_count_mean,
        "measured_violation_count_mean": measured_violation_count_mean,
        "actual_violations_average_ratio": actual_violations_average_ratio,
        "measured_violations_average_ratio": measured_violations_average_ratio,
        "max_actual_violation": max_actual_violation,
        "max_measured_violation": max_measured_violation,
        "TF": TF,
        "DT": DT,
        "initial_state": initial_state,
        "wall_x": wall_x,
        "base_key_seed": base_key_seed,
    }

    # Save the data to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f"Sim results saved to {file_path}")


