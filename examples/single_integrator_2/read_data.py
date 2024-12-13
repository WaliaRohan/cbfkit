




runs = 5
    start_run = 1
    end_run = 100

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

        sim_start = time.time()
        x, u, z, p, dkeys, dvalues, measurements = simulate(key=key,
                                                            initial_state=initial_state,
                                                            wall_x=wall_x,
                                                            DT=DT, TF=TF,
                                                            class_k_gain=class_k_gain)
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