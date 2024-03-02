import time
import numpy as np
import matplotlib.pyplot as plt
import os

def get_pedestrian_speeds(scenario, iterations):
    """
    @return: desired_speeds, actual_speeds
    """
    ped_desired_speeds = []
    ped_speeds = []
    ped_pos = []

    for pedestrian in scenario.pedestrians:
        speed = (scenario.target_distance_grids[pedestrian.start_pos] -
                 scenario.target_distance_grids[pedestrian.position]) / iterations
        ped_desired_speeds.append(pedestrian.desired_speed)
        ped_speeds.append(speed)
        ped_pos.append(pedestrian.start_pos)

    for pedestrian, finish_it in scenario.pedestrians_in_goal:
        speed = (scenario.target_distance_grids[pedestrian.start_pos] -
                 scenario.target_distance_grids[pedestrian.position]) / finish_it
        ped_desired_speeds.append(pedestrian.desired_speed)
        ped_speeds.append(speed)
        ped_pos.append(pedestrian.start_pos)

    ped_desired_speeds = np.asarray(ped_desired_speeds, dtype=float)
    ped_speeds = np.asarray(ped_speeds, dtype=float)

    # Sort the indices based on desired speeds
    inds = np.argsort(ped_desired_speeds)
    x = ped_desired_speeds[inds]
    y = ped_speeds[inds]
    return x,y,ped_pos

def make_2D_hist_of_speed(scenario, iterations):
    x,y,_ = get_pedestrian_speeds(scenario,iterations)
    make_2d_hist(x, y, 'ped desired speeds', 'ped actual speeds')

def make_2d_hist(x, y, x_label, y_label, plot_line = True):
    plt.clf()
    ax1 = plt.axes()
    hb = ax1.hexbin(x, y, gridsize=20, cmap='rainbow')
    
    ax1.plot(x, y, 'k-', label='Data')  #comment to not plotting data line

    # Set the title, x-label, and y-label as before
    ax1.set_title('hexbin')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid()

    # Create a legend
    ax1.legend(loc='upper right', frameon=False)

    # Create a color bar for the pedestrian density
    cb = plt.colorbar(hb, ax=ax1)
    cb.set_label('Pedestrian density')  # Add label to the color bar

    if not os.path.exists("images"):
        os.makedirs("images")

    plt.savefig(f"images/{time.strftime('%Y%m%d-%H%M%S')}")
    plt.show()