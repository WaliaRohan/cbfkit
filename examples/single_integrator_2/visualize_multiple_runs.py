import os
import sys

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add 'src/cbfkit' to the Python path
cbfkit_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(cbfkit_path)
print(current_dir, "------", cbfkit_path)

# Add the parent directory to the system path
sys.path.append(os.path.join(current_dir, "src"))

def visualize():
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
    ax.plot(x, y, label='True Trajectory')
    measurements = np.array(measurements)

    if len(x) == len(measurements):
        # Plot measurements
        ax.plot(measurements, y, label='Measured Trajectory', linewidth=0.5)
        # Optionally add a legend to differentiate the data
        ax.legend()
        # ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
        # verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))


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

if __name__=="__main__":
    
    runs = 10
    initial_state = jnp.array([0.0])
    wall_x = 9.0

    base_key = random.PRNGKey(0)  # Starting key
    keys = random.split(base_key, runs)  # Generate 1000 unique keys

    results = []

    violations = []
    violation_ratios = []

    for key in keys:
        x, u, z, p, dkeys, dvalues, measurements = simulate(key=key,
                                                            initial_state=initial_state,
                                                            wall_x=wall_x)
        
        # Convert to numpy
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        u = np.array(u) if not isinstance(u, np.ndarray) else u
        z = np.array(z) if not isinstance(z, np.ndarray) else z
        p = np.array(p) if not isinstance(p, np.ndarray) else p
        measurements = np.array(measurements) if not isinstance(measurements, np.ndarray) else measurements
        
        run_violations = jnp.sum(x > wall_x).item()  # Count how many values in x are greater than wall_x
        violations.append(run_violations)  # Append the count to the violations list

        run_ratio = run_violations / len(x) if len(x) > 0 else 0  # Avoid division by zero
        violation_ratios.append(run_ratio)  # Append the ratio to the ratios list

        results.append({
            "x": x,
            "u": u,
            "z": z,
            "p": p,
            "dkeys": dkeys,
            "dvalues": dvalues,
            "measurements": measurements
        })

    # Calculate the average ratio across all runs
    average_ratio = np.mean(violation_ratios) if violation_ratios else 0

    # Print the results
    for i, count in enumerate(violations):
        print(f"Run {i + 1}: Number of violations (values of x greater than {wall_x}): {count}")
    
    print(f"Average ratio of violations to length of x across all runs: {average_ratio:.2f}")