import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import bifurcations as bif


def plot_logistic_map(interval, accuracy, reps, numtoplot, warmup):
    """
    Plots the bifurcation diagram of the logistic map.

    Parameters:
    interval (tuple): The range of r values to plot.
    accuracy (float): The step size for the r values.
    reps (int): The number of iterations to perform for each r value.
    numtoplot (int): The number of final iterations to plot for each r value.
    warmup (int): The number of initial iterations to discard.

    Returns:
    None
    """
    fig, biax = plt.subplots()
    fig.set_size_inches(14, 7)

    # Map the color and label to the range of r values
    color_map = {
        "g.": {"condition": lambda r: 0 <= r <= 1, "label": "0 <= r <= 1"},
        "b.": {"condition": lambda r: 1 < r <= 2, "label": "1 < r <= 2"},
        "r.": {"condition": lambda r: 2 < r <= 3, "label": "2 < r <= 3"},
        "c.": {"condition": lambda r: 3 < r <= 4, "label": "3 < r <= 4"}
    }

    for r in np.arange(interval[0], interval[1], accuracy):
        lims = bif.simulate_logistic_map(r, np.random.rand(), warmup + reps)

        # Plot for each color
        for color, info in color_map.items():
            if info["condition"](r):
                biax.plot([r] * numtoplot, lims[-numtoplot:], color, markersize=0.02)
                break

    # Create custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=color_info["label"], 
                              markerfacecolor=color[0], markersize=10) for color, color_info in color_map.items()]
    biax.legend(handles=legend_elements)

    biax.set(xlabel="r", ylabel="x", title="Logistic Map Bifurcation Diagram")
    plt.savefig("logistic_map.png", dpi=150)
    plt.show()


def plot_trajectory_lorenz(trajectory, linewidth = 0.05):
    """
    Plots a single trajectory of the Lorenz Attractor.

    Parameters:
    trajectory (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the trajectory.

    Returns:
    None
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(*trajectory.T, lw=linewidth, color='blue')  # Actual plot with lw=0.05
    line, = ax.plot([], [], [], label='Trajectory', lw=2, color='blue')  # Dummy plot for legend with lw=2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    ax.legend()
    plt.savefig("trajectory_lorentz.png", dpi=150)
    plt.show()


def plot_trajectory_lorenz_xz(trajectory, linewidth = 0.1):
    """
    Plots a single trajectory of the Lorenz Attractor on the x-z plane.

    Parameters:
    trajectory (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the trajectory.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trajectory[:, 0], trajectory[:, 2], lw=linewidth, color='blue')  # Actual plot with lw=0.1
    line, = ax.plot([], [], label='Trajectory', lw=2, color='blue')  # Dummy plot for legend with lw=2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.set_title("Lorenz Attractor (X-Z plane)")
    ax.legend()
    plt.savefig("trajectory_lorentz_xz.png", dpi=150)
    plt.show()


def plot_multrajectories_lorenz(trajectory_original, trajectory_perturbed, linewidth = 0.05):
    """
    Plots two trajectories of the Lorenz Attractor.

    Parameters:
    trajectory_original (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the original trajectory.
    trajectory_perturbed (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the perturbed trajectory.

    Returns:
    None
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(*trajectory_original.T, lw=linewidth, color='blue')  # Actual plot with lw=0.05
    ax.plot(*trajectory_perturbed.T, lw=linewidth, color='orange')  # Actual plot with lw=0.05
    line1, = ax.plot([], [], [], label='Original Trajectory', lw=2, color='blue')  # Dummy plot for legend with lw=2
    line2, = ax.plot([], [], [], label='Perturbed Trajectory', lw=2, color='orange')  # Dummy plot for legend with lw=2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor - Trajectories")
    ax.legend()
    plt.savefig("trajectory_lorentz_mul.png", dpi=150)
    plt.show()

def plot_multrajectories_lorenz_xz(trajectory_original, trajectory_perturbed, linewidth = 0.1):
    """
    Plots two trajectories of the Lorenz Attractor on the x-z plane.

    Parameters:
    trajectory_original (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the original trajectory.
    trajectory_perturbed (numpy.ndarray): A 2D array where each row is a point (x, y, z) on the perturbed trajectory.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trajectory_original[:, 0], trajectory_original[:, 2], lw=linewidth, color='blue')  # Actual plot with lw=0.1
    ax.plot(trajectory_perturbed[:, 0], trajectory_perturbed[:, 2], lw=linewidth, color='orange')  # Actual plot with lw=0.1
    line1, = ax.plot([], [], label='Original Trajectory', lw=2, color='blue')  # Dummy plot for legend with lw=2
    line2, = ax.plot([], [], label='Perturbed Trajectory', lw=2, color='orange')  # Dummy plot for legend with lw=2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.set_title("Lorenz Attractor - Trajectories (X-Z plane)")
    ax.legend()
    plt.savefig("trajectory_lorentz_mul_xz.png", dpi=150)
    plt.show()



def plot_distance_lorenz(num_steps, dt, distance):
    """
    Plots the Euclidean distance between two trajectories of the Lorenz Attractor over time.

    Parameters:
    num_steps (int): The number of time steps.
    dt (float): The time step size.
    distance (numpy.ndarray): A 1D array of the Euclidean distances at each time step.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(num_steps) * dt, distance, label='Distance')
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Difference between Trajectories Over Time")
    ax.legend()
    plt.savefig("distance_trajectories.png", dpi=150)
    plt.show()