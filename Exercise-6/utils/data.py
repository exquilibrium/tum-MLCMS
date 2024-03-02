from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os
parent_dir = os.path.dirname(os.path.realpath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
# Define the directories where the bottleneck and corridor data files are stored
bottleneck_folder = parent_parent_dir / Path("Bottleneck_Data")
corridor_folder = parent_parent_dir / Path("Corridor_Data")

# Create lists of all filepaths in the bottleneck and corridor directories
bottleneck_files = [x for x in bottleneck_folder.iterdir()]
corridor_files = [x for x in corridor_folder.iterdir()]


def makedata(file_name):
    """
    Reads a file and converts its content into a numpy array.
    
    Parameters:
    file_name (str): The name of the file to read.

    Returns:
    numpy.ndarray: The data from the file as a numpy array.
    """
    file = open(file_name)
    data = []
    for i in file:
        data.append([float(x) for x in i.split(" ")])
    return np.array(data)


def transform(data):
    """
    Normalizes the last two columns of a numpy array.
    
    Parameters:
    data (numpy.ndarray): The data to normalize.

    Returns:
    numpy.ndarray: The normalized data.
    """
    # Delete z position from data
    data = data[:, :4]
    # Convert x y positions to meters
    data[:, 2:] = data[:, 2:]/100
    
    return data

def getpedpos(datapre, everyxth = 1, usefuture = 1, usepast = False):
    """
    A more general method for frame extraction, was abandoned for the sake of data processing unification.
    """
    peds = np.unique(datapre[:, 0])
    pedpos = {}
    for idped in peds:
        d = {}
        for ind in np.where(datapre[:, 0] == idped)[0]:
            t = datapre[ind, 1]
            if t % everyxth != 0:
                continue
            pos = datapre[ind, 2:]
            x = usefuture
            if ind + x < datapre.shape[0]:
                idind1 = datapre[ind + x, 0]
                if idind1 == idped:
                    assert datapre[ind + x, 1] == t + x, f'{everyxth} {x} {datapre[ind + x, 1]} {datapre[ind, 1] + x}'
                    pos_tpx = datapre[ind + x, 2:]
                    if usepast:
                        if ind - x < 0 or datapre[ind-x, 1] != t-x:
                            continue
                        otherpos = datapre[ind-x:ind + x + 1, 2:]
                        speed = np.average(np.sqrt(np.sum(np.square(otherpos[1:] - otherpos[:-1]), 1)))
                    else:
                        speed = np.sqrt(np.sum(np.square(pos-pos_tpx)))/x
                    d[t] = (pos, pos_tpx+((pos-pos_tpx)*16)/x, speed*16)
        pedpos[idped] = d
    return pedpos

def makedatafrompedpos(pedpos, k, args, removefewneighbors = True, usedids = None, usedidsneighbors = None):
    """
    A more general method for data processing, was abandoned for the sake of data processing unification.
    """
    data = []
    targets = []
    if usedids is None:
        usedids = pedpos.keys()
    if usedidsneighbors is None:
        usedidsneighbors = pedpos.keys()
    for idped in usedids:
        for t in pedpos[idped]:
            pos, pos_tp1, speed = pedpos[idped][t]
            otherposs = np.array([pedpos[x][t][:2] for x in usedidsneighbors if t in pedpos[x]])
            if (removefewneighbors and otherposs.shape[0] < k) or otherposs.shape[0] < 3:
                continue
            reltivepos = otherposs[:, 0, :] - pos
            velocity = pos - pos_tp1
            inds = np.argsort(np.linalg.norm(reltivepos, axis=1), 0)[1:k+1]
            vs = reltivepos[inds, :]
            if args.use_rel_vel:
                relativevelocities = (otherposs[:, 0, :] - otherposs[:, 1, :]) - velocity
                vs = np.concatenate([vs,relativevelocities[inds, :]], 1)
            if vs.shape[0] < k:
                vs = np.concatenate([vs, np.zeros((k - vs.shape[0], vs.shape[1]))], 0)
            data.append(np.append(vs.flatten(), np.mean(np.linalg.norm(reltivepos)))
                                       if args.use_mean_dist_spacing else vs.flatten())
            targets.append(speed)
    return np.stack(data, 0), np.stack(targets, 0)

def visualize_trajectory(files, name):
    """
    Visualizes trajectory data from a set of files.

    Parameters:
    files (list): A list of file objects.
    name (str): A string to be printed as the title of each plot.
    """
    # Loop over each file in the list
    for x in files:
        data = makedata(x)
        # Get the unique IDs and times from the data
        ids = np.unique(data[:, 0])
        # times = np.unique(data[:, 1])  # unused
        
        # Plot the positions for each ID
        for ped_id in ids:
            pos = data[np.where(data[:, 0] == ped_id)]
            # The color is set to cyan for all points
            plt.scatter(pos[:, 2], pos[:, 3], 10, np.broadcast_to(np.array([[0.5, 1, 1]]),(pos.shape[0], 3)))
        
        # Highlight the first two IDs by plotting their positions with a color gradient
        for ped_id in ids[:2]:
            pos = data[np.where(data[:, 0] == ped_id)]
            # The color varies from black to white depending on the position in the array
            plt.scatter(pos[:, 2], pos[:, 3], 10,
                        np.stack([np.arange(0, pos.shape[0]) for _ in range(3)]).T/pos.shape[0],
                        label=f'Highlighted ID {ped_id}')

        # Add labels and a title to the plot
        file_name = str(x).split("/")[1].split(".")[0]  # "Bottleneck '/' uo-180-120 '.' txt"
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{name} {file_name} Data Visualization')
        plt.legend()
        plt.show()


