import numpy as np
from scipy.integrate import solve_ivp

def logistic_map(r, x):
    """
    Computes the next value in the logistic map sequence.

    Parameters:
    r (float): The growth rate parameter.
    x (float): The current value in the sequence.

    Returns:
    float: The next value in the sequence.
    """
    return r * x * (1 - x)

def simulate_logistic_map(r, x0, num_steps):
    """
    Simulates the logistic map for a given number of steps.

    Parameters:
    r (float): The growth rate parameter.
    x0 (float): The initial value in the sequence.
    num_steps (int): The number of steps to simulate.

    Returns:
    numpy.ndarray: An array of the simulated values.
    """
    x = np.empty(num_steps)
    x[0] = x0
    for i in range(1, num_steps):
        x[i] = logistic_map(r, x[i-1])
    return x

def lorenz(t, xyz, sigma, rho, beta):
    """
    Computes the derivatives for the Lorenz system.

    Parameters:
    t (float): The current time.
    xyz (list): The current values of x, y, and z.
    sigma, rho, beta (float): The parameters of the Lorenz system.

    Returns:
    list: The derivatives dx/dt, dy/dt, and dz/dt.
    """
    x, y, z = xyz
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return [x_dot, y_dot, z_dot]

def simulate_lorenz(initial_values, Tend, dt, sigma=10, rho=28, beta=8/3):
    """
    Simulates the Lorenz system for a given time period.

    Parameters:
    initial_values (list): The initial values of x, y, and z.
    Tend (float): The end time for the simulation.
    dt (float): The time step size.
    sigma, rho, beta (float): The parameters of the Lorenz system.

    Returns:
    numpy.ndarray: A 2D array where each row is a point (x, y, z) on the trajectory.
    """
    t = np.arange(0, Tend, dt)
    sol = solve_ivp(lorenz, (0, Tend), initial_values, args=(sigma, rho, beta), t_eval=t)
    return sol.y.T