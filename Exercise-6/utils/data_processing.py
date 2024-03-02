import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from kb.weidmann_fd import *

# Create lists of all filepaths in the bottleneck and corridor directories
bottleneck_files = [x for x in Path("Bottleneck_Data").iterdir()]
corridor_files = [x for x in Path("Corridor_Data").iterdir()]


def load_file(file_name):
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


def load_directory(files):
    """
    Load data from a folder into a dictionary

    Parameters:
        files (list): List of file names

    Returns:
        dictionary: Filenames are the keys
    """
    data_dict = {}
    for file in files:
        raw_data = load_file(file)
        filename = str(file).split("/")[1].split(".")[0]  # "Bottleneck '/' uo-180-120 '.' txt"
        data_dict[filename] = raw_data
    return dict(sorted(data_dict.items()))


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
    data[:, 2:] = data[:, 2:] / 100

    return data


def remove_oob_bottleneck(data):
    """
    Removes data entries that have negative values

    Parameters:
    data (numpy.ndarray): The data to normalize.

    Returns:
    numpy.ndarray: The normalized data.
    """
    data = data[np.where(data[:, 2] >= 0.0)]
    data = data[np.where(data[:, 2] <= 1.8)]
    data = data[np.where(data[:, 3] >= 0.0)]
    data = data[np.where(data[:, 3] <= 8.0)]
    return data


def remove_oob_corridor(data):
    """
    Removes data entries that have negative values

    Parameters:
    data (numpy.ndarray): The data to normalize.

    Returns:
    numpy.ndarray: The normalized data.
    """
    data = data[np.where(data[:, 2] >= 0.0)]
    data = data[np.where(data[:, 2] <= 1.8)]
    data = data[np.where(data[:, 3] >= 0.0)]
    data = data[np.where(data[:, 3] <= 6.0)]
    return data


def calculate_spacing(dataset, k):
    """
    Calculate mean spacing s_K for each pedestrian and each frame

    Parameters:
        dataset (np.array): Dataset
        k (int): Number of nearest neighbours

    Returns:
        ms (np.array): Vector of mean spacing
    """
    ms = []
    knn = NearestNeighbors(n_neighbors=k)
    for d in dataset:
        # Filter by frame id
        frame_id = d[1]
        frames = dataset[np.where(dataset[:, 1] == frame_id)]
        # Get x y coordinates
        d_xy = d[2:4]
        frames_xy = frames[:, 2:4]
        # Find nearest neighbours
        if len(frames) > k:
            knn.fit(frames_xy)
            dist, _ = knn.kneighbors([d_xy], return_distance=True)
            ms.append(np.mean(dist))
        else:
            # Calculate nearest neighbours even if nn <= k
            ms.append(spacing(d_xy, frames_xy))
    return np.array(ms)


def calculate_speeds(dataset, old=False):
    """
    Calculate speeds v for each pedestrian and each frame

    Parameters:
        dataset (np.array): Dataset
        old (bool, optional): Whether to use the old speed function

    Returns:
        speeds (np.array): Speeds
    """
    speeds = np.array([])
    # Filter by pedestrian id
    ids = np.unique(dataset[:, 0])
    for ped_id in ids:
        # Get positions
        frames = dataset[np.where(dataset[:, 0] == ped_id)]
        next_frames = frames[:, 2:4]  # 1 2 3
        curr_frames = np.vstack([next_frames[0], next_frames[:-1]])  # 1 1 2
        # Calculate speed
        speeds = np.concatenate((speeds, speed_old(next_frames, curr_frames) if old
                                    else speed(next_frames, curr_frames)))
    return speeds


def calculate_density(dataset, bottleneck):
    """
    Calculate density D for each pedestrian and each frame

    Parameters:
        dataset (np.ndarray): Dataset
        bottleneck (Bool) : Whether bottleneck or corridor data is used
    """
    bottleneck_area = 8.0 * 1.8
    corridor_area = 6.0 * 1.8
    # Filter by pedestrian id
    ds = []
    density = []
    density_dict = {}
    ids = np.unique(dataset[:, 1])
    for frame_id in ids:
        # Get positions
        frames = dataset[np.where(dataset[:, 1] == frame_id)]
        # Calculate density
        d = frames.shape[0] / (bottleneck_area if bottleneck else corridor_area)
        density.append(d)
        density_dict[frame_id] = d
    for d in dataset:
        ds.append(density_dict[d[1]])
    return np.array(ds), np.array(density)


def write_dataset(data, filename):
    """
    Write dataset into directory for cleaned data

    Parameters:
        data (np.ndarray): Cleaned dataset
        filename (str): Filename
    """
    f = open(filename + '-clean.txt', 'w')
    for line in data:
        x = str(int(line[0])) + ' ' + str(int(line[1]))
        for i in range(len(line) - 2):
            x += ' ' + '%.3f' % line[i + 2]
        f.write(x + '\n')
