import numpy as np


def w(sk, v0, time, l):
    """
    Weidmann Fundamental Diagram (1994) according to Tordeux's paper
    Non-linear function to calculate pedestrian speed

    Parameters:
        sk: mean spacing
        v0: speed in free situation
        time: time of following gap between neighbours
        l: physical size of the pedestrian

    Returns:
        v: pedestrian speed
    """
    return v0 * (1 - np.exp((l-sk)/(v0 * time)))


def weidmann(dens, v0):
    """
    Weidmann Fundamental Diagram (1993)
    Non-linear function to calculate pedestrian speed based on density

    Parameters:
        dens: density
        v0: speed in free situation

    Returns:
        v: pedestrian speed
    """
    gamma = 1.913
    Dmax = 5.4
    return v0 * (1 - np.exp(-gamma * (1/dens - 1/Dmax)))


"""
Helper functions for Weidmann Fundamental Diagram
"""
def w_wrap(sk, v0, time):
    return w(sk, v0, time, 0.625)
def w_bottleneck(sk, v0):
    return w(sk, v0, 0.49, 0.625)
def w_corridor(sk, v0):
    return w(sk, v0, 0.85, 0.625)
def w_bottleneck_v0(sk, time):
    return w(sk, 1.308, time, 0.625)
def w_corridor_v0(sk, time):
    return w(sk, 0.819, time, 0.625)


def spacing(x, positions):
    """
    Calculates the mean spacing s_K given an array of positions and point x

    Parameters:
        x (np.array): position pedestrian
        positions (np.array): 2D array of K nearest pedestrian positions
    """
    a = np.array([x] * positions.shape[0])
    dist = np.linalg.norm(a - positions)
    return np.sum(dist) / positions.shape[0]


def speed_old(pos0, pos1):
    """
    Calculates the speed v in m/s

    Parameters:
        pos0 (np.array): Position of pedestrian
        pos1 (np.array): Position of pedestrian in the next frame

    Returns:
        speed (float):
    """
    # Multiply by 16 to scale back up to 1s
    return np.linalg.norm(pos0-pos1, axis=1) * 16


def speed(pos0, pos1):
    """
    Calculates the speed v in m/s

    Parameters:
        pos0 (np.array): Position of pedestrian
        pos1 (np.array): Position of pedestrian in the next frame

    Returns:
        speed (float):
    """
    if pos0.shape[0] > 16:
        n = []
        for i in range(pos0.shape[0]-16):
            x = np.linalg.norm(pos0[i] - pos1[i+16])
            n.append(x)
        n = np.array(n)
        m = np.mean(n)
        n = np.concatenate((np.array([m] * 8), n, np.array([m] * 8)))
    else:
        n = speed_old(pos0, pos1)
    return n


def mse(f, x, y, opt):
    """
    Calculates the MSE between the predicted and original data

    Parameters:
        f: Weidmann function
        x (np.array): Input
        y (np.array): Labels
        opt (np.array): Optimized parameters

    Returns:
        mse (float): The MSE and standard deviation of the data
    """
    temp = np.square(f(x, *opt[0]) - y)
    return np.mean(temp), np.std(temp)

