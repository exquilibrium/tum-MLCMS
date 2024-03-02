import matplotlib.pyplot as plt
import numpy as np


def plot_ped(d):
    traj = d.reshape(-1, 2)
    plt.scatter(traj[:, 0], traj[:, 1])


def mean(data):
    return np.expand_dims(np.mean(data, 0), 0)
