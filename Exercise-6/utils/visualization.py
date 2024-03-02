import numpy as np
from matplotlib import pyplot as plt


def visualize_all_trajectories(data_dictionary, scenario_name):
    """
    Visualize all trajectories in the data dictionary

    Parameters:
        data_dictionary: Dictionary of data to be visualized
        scenario_name: Name of the test scenario
    """
    for key in data_dictionary:
        visualize_trajectory(data_dictionary[key], scenario_name, key)


def visualize_trajectory(data, scenario, filename):
    """
    Visualizes trajectory data from a set of files.

    Parameters:
        files (list): A list of file objects.
        name (str): A string to be printed as the title of each plot.
    """
    # Get the unique IDs and times from the data
    ids = np.unique(data[:, 0])

    # Plot the positions for each ID
    for ped_id in ids:
        pos = data[np.where(data[:, 0] == ped_id)]
        # The color is set to cyan for all points
        plt.scatter(pos[:, 2], pos[:, 3], 10, np.broadcast_to(np.array([[0.5, 1, 1]]), (pos.shape[0], 3)))

    # Highlight the first two IDs by plotting their positions with a color gradient
    for ped_id in ids[:2]:
        pos = data[np.where(data[:, 0] == ped_id)]
        # The color varies from black to white depending on the position in the array
        plt.scatter(pos[:, 2], pos[:, 3], 10,
                    np.stack([np.arange(0, pos.shape[0]) for _ in range(3)]).T / pos.shape[0],
                    label=f'Highlighted ID {ped_id}')

    # Add labels and a title to the plot
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'{scenario} {filename} Data Visualization')
    plt.legend()
    plt.show()



def plot_z_coordinate_vs_time(data_dict, title, num_ids):
    """
    Plots the Z coordinate against the frame number for a number of random IDs in each file in a data dictionary.

    Parameters:
    data_dict (dict): A dictionary where the keys are file names and the values are numpy arrays of data.
    title (str): The title of the plot.
    num_ids (int): The number of random IDs to plot.
    """
    # Loop over each file in the data dictionary
    for filename in data_dict:
        
        data = data_dict[filename]
        
        # Get the unique IDs
        unique_ids = np.unique(data[:, 0])
        
        # Select a number of random IDs
        random_ids = np.random.choice(unique_ids, num_ids, replace=False)
        
        # Loop over each random ID
        for id in random_ids:
            # Get the data for this ID
            id_data = data[data[:, 0] == id]
            
            # Get the frame numbers and Z coordinates
            frames = id_data[:, 1]
            z_coords = id_data[:, 4]
            
            # Plot the Z coordinates against the frame numbers
            plt.plot(frames, z_coords, label=f'ID {id} in {filename}')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Z Coordinate')
    plt.title(title)
    plt.show()


def visualize_weidmann(w, x, y, opt, x_label, weid=False):
    p = 3000
    if np.shape(x)[0] > p:
        idx = np.random.choice(np.shape(x)[0], p)
        x = x[idx]
        y = y[idx]
    plt.scatter(x, y, facecolors='none', edgecolors='r', marker='^', alpha=0.1)
    # Sort x
    x = np.unique(x)
    if np.shape(opt[0])[0] == 2:
        plt.plot(x, w(x, *opt[0]), 'b--', label='fit: v0=%5.3f, T=%5.3f' % tuple(opt[0]))
    elif weid:
        plt.plot(x, w(x, *opt[0]), 'b--', label='fit: T=%5.3f' % tuple(opt[0]))
    else:
        plt.plot(x, w(x, *opt[0]), 'b--', label='fit: v0=%5.3f' % tuple(opt[0]))
    plt.xlabel(x_label)
    plt.ylabel('Speed')
    plt.legend()
    plt.show()



