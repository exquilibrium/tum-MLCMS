import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist



def load_dataset(filename, sort=True, separator=' '):
    """
    Loads every line of a file <filename>
    into a 2d-array (by splicing that string)
    resulting in shape:
    (<#lines>, <#elements in the line>)

    :param filename: Filename
    :param sort: Sort the array by the 0th column
    :param separator: String used to recognize splices
    :return: Data-set as 2D-array of shape (<#lines>, <#elements in the line>)
    """
    lines = list(str.strip(x) for x in open(filename, "r").readlines())
    data = np.asarray(list(tuple(float(y) for y in x.split(separator)) for x in lines if len(x) > 0))
    if sort:
        return data[data[:, 0].argsort()]
    return data


def prepare_data(data, dim_x, dim_y):
    """
    A helper function to get feature and label matrix X, Y

    :param data: Shape (N, dim_x+dim_y), dim_x-dimensional feature x, dim_y-dimensional label y
    :param dim_x: Dimension of feature x
    :param dim_y: dimension of label y
    :return: x, y or shape (N,dim_x), (N, dim_y)
    """
    # For 1-dimensional features x and labels y
    # x = data[:, 0]  # Get 1st column of data, replace 0 with n if x is n-dimensional
    # y = data[:, 1]  # Get 1st column of data, replace 1 with n if x is n-dimensional

    # For n-dimensional features x and d-dimensional labels y
    x = np.asarray([data[:, i] for i in range(dim_x)]).T  # Get dim_x feature columns from data
    y = np.asarray([data[:, dim_x+i] for i in range(dim_y)]).T  # Get dim_y label columns from data
    return x, y


###################################################################################################
# Functions for Task 1
###################################################################################################

def expand_data(funcs, m):
    """
    A higher order function that applies a list of lambdas
    on every x in data m of shape (N,1). This expands x to
    high dimensional x in R^n and m to shape (N,n).
    n is the number of lambdas.

    :param funcs: list of lambdas
    :param m: Data of shape (N,1)
    :return: Data of shape (N,n)
    """
    r = []
    for i in range(len(funcs)):
        r.append(funcs[i](m.T[0]))
    return np.asarray(r).T


def lstsq_direct(x_mat, y):
    """
    A helper function.
    Implementation of the least squares closed-form solutions.
    Do not use! It's only for demonstration!

    :param x_mat: Features
    :param y: Targets
    :return: Weight-matrix A
    """
    return np.dot((np.dot(np.linalg.inv(np.dot(x_mat.T, x_mat)), x_mat.T)), y)


def linear_lstsq(data, bias=False, cond=1.0, direct=False, exec_t=False, dim_x=1, dim_y=1, lambdas=[], lin=False):
    """
    Approximates dataset using a linear function.
    Tutorial: https://jiffyclub.github.io/scipy/tutorial/linalg.html

    :param data: Shape (N, n+d), n-dimensional feature x, d-dimensional label y
    :param bias: Include a bias for linear function
    :param cond: Cutoff for ‘small’ singular values; used to determine effective rank of "a".
                 Singular values smaller than cond * largest_singular_value are considered zero.
    :param direct: Use our implementation of linear least squares. NOT recommended!
    :param exec_t: Print execution time
    :param dim_x: Dimension of features x
    :param dim_y: Dimension of labels y
    :param lambdas: List of functions to be applied on x_i
    :param lin: Use lin-space x coordinates for lambdas
    :return: x, y, y2 coordinates as float list and the matrix c_mat
    """
    # Prepare data
    x, y = prepare_data(data, dim_x, dim_y)  # Get feature, label columns of data
    length = len(lambdas)
    if bias and length == 0:  # Transform shape of x to matrix
        ones = np.full(np.shape(x)[0], 1)
        x = [e for e in x.T]
        x.append(ones)
        x = np.asarray(x).T  # Shape (1000, dim_x+1)
    if length > 0 and dim_x == 1:
        x = expand_data(lambdas, x)

    # Least-Squares
    start = time.time()
    c = lstsq_direct(x, y) if direct else sp.linalg.lstsq(x, y, cond=cond)[0]
    end = time.time()
    if exec_t:  # Print execution time
        print(f'Execution time of {"our" if direct else "lib"}: {end-start}')

    x2 = x
    if length > 0 and dim_x == 1 and lin:
        x2 = np.asarray(np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), len(x))[:, np.newaxis])
        x2 = expand_data(lambdas, x2)

    y2 = np.matmul(x2, c)
    return x, y, x2, y2, c


def rbf_gauss(x, xl, epsilon, sq):
    """
    A helper function.
    Calculates a gaussian rbf.

    :param x: x coordinate
    :param xl: center of basis function (where it obtains maximum value 1)
    :param epsilon: Bandwidth for xl
    :param sq: Square the bandwidth
    :return: Gaussian thingy
    """
    a = -np.square(np.linalg.norm(xl - x))
    b = epsilon**2 if sq else epsilon
    return np.exp(a/b)


def rbf_lstsq(data, L=100, epsilon=1.0, sq=True, domain=0, dim_x=1, dim_y=1, lib=False, direct=False,
              cond=1.0, func='gaussian', norm='euclidean'):
    """
    Approximate data using radial basis function.
    Tutorial: https://scipy.github.io/old-wiki/pages/Cookbook/RadialBasisFunctions.html

    :param data: Shape (N, n+d), n-dimensional feature x, d-dimensional label y
    :param L: Number of (non-linear) basis functions
    :param epsilon: Bandwidth for xl
    :param sq: Square the bandwidth epsilon
    :param domain: Increases the domain size additively in both direction
    :param dim_x: Dimension of features x
    :param dim_y: Dimension of labels y
    :param lib: Use library implementation of Radial Basis Functions
    :param direct: Use our implementation of linear least squares. NOT recommended!
    :param cond: Cutoff for ‘small’ singular values; used to determine effective rank of "a".
                 Singular values smaller than cond * largest_singular_value are considered zero.
    :param func: The radial basis function; default is 'gaussian', for other basis function
                 see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    :param norm: The distance norm; default is 'sqeuclidean', for other distance metrics
                 see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    :return: x, y, y2 coordinates as float list and the matrix c_mat
    """
    # Prepare data
    x, y = prepare_data(data, dim_x, dim_y)  # Get feature, label columns of data
    domain = 0 if domain < 0 else domain
    x2 = [np.linspace(np.min(x)-domain, np.max(x)+domain, L) for _ in range(np.shape(x[0])[0])]  # Shape (n=1, L=100)
    x2 = np.asarray(x2).T  # Define L center points # Shape (L=100, n=1)

    # Radial basis function
    if lib:
        # RBF
        _rbf = sp.interpolate.Rbf(x, y, function=func, epsilon=epsilon if sq else np.sqrt(epsilon), norm=norm)
        y2 = _rbf(x2)
        c = None
    else:
        # Our implementation
        phi = np.asarray([[rbf_gauss(xl, xi, epsilon, sq) for xl in x2] for xi in x])  # Shape (1000, 100)
        c = lstsq_direct(phi, y) if direct else sp.linalg.lstsq(phi, y, cond=cond)[0]
        y2 = np.matmul(np.asarray([[rbf_gauss(xl, xi, epsilon, sq) for xl in x2] for xi in x2]), c)  # Shape (100, 1)
    return x, y, x2, y2, c


def plot_comparison(x, y, x2, y2, line=True, title='Data fitting'):
    """
    Plot data-set and approximated function
    Only tested for 2-dimensional data, i.e. x,y in R.

    :param x: x-coordinate
    :param y: y-coordinate of data-set
    :param x2: Corresponding x coordinate of approximation
    :param y2: Approximated y coordinate
    :param line: Visualize plot line instead of scatter plot for x2,y2; requires sorted data
    :param title: Title of plot
    :return: None
    """
    # Get first dimension of features and labels
    x = x[:, 0]
    x2 = x2[:, 0]
    y = y[:, 0]
    y2 = y2[:, 0]

    # Plot data-set as 'x'
    plt.plot(x, y, 'x', alpha=0.5)
    # Plot approximated function
    if line:
        plt.plot(x2, y2, c='orange')
    else:
        plt.scatter(x2, y2, s=3, c='orange', zorder=3)
    # Other stuff
    plt.title(title)
    plt.show()


###################################################################################################
# Functions for Task 4
###################################################################################################

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


def plot_trajectory_lorenz(trajectory, delay, linewidth = 0.05):
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
    ax.set_title("Lorenz Attractor (Delay = {})".format(delay))
    ax.legend()
    plt.savefig("trajectory_lorentz.png", dpi=150)
    plt.show()


def plot_trajectory_lorenz_xz(trajectory, delay, linewidth = 0.1):
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
    ax.set_title("Lorenz Attractor (X-Z plane) (Delay = {})".format(delay))
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
    line2, = ax.plot([], [], [], label='Approximated Trajectory', lw=2, color='orange')  # Dummy plot for legend with lw=2
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
    line2, = ax.plot([], [], label='Approximated Trajectory', lw=2, color='orange')  # Dummy plot for legend with lw=2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.set_title("Lorenz Attractor - Trajectories (X-Z plane)")
    ax.legend()
    plt.savefig("trajectory_lorentz_mul_xz.png", dpi=150)
    plt.show()

def best_delay(trajectory, x):
    """
    Determines the best delay for the Lorenz system by minimizing the Euclidean distance 
    between the original and delayed trajectories.

    Parameters:
    trajectory (numpy.ndarray): The original trajectory of the Lorenz system.
    x (numpy.ndarray): The x-coordinate of the Lorenz system.

    Returns:
    best_delay (int): The delay that minimizes the Euclidean distance between the original 
                      and delayed trajectories.
    distances (list): The minimum Euclidean distance for each delay.

    """
  
    min_distance = float('inf')
    best_delay = None

    # Initialize a list to store the minimum distance for each delay
    distances = []

    # Loop over possible delays
    for delay in range(1, 100): 
        # Create delayed versions of the x-coordinate
        x_delayed1 = np.roll(x, -delay)
        x_delayed2 = np.roll(x, -2*delay)

        # Create the new trajectory
        new_trajectory = np.column_stack((x, x_delayed1, x_delayed2))

        # Compute the distance between the original and new trajectories
        distance = cdist(trajectory, new_trajectory, 'euclidean').min()

        distances.append(distance)

        if distance < min_distance:
            min_distance = distance
            best_delay = delay

    # Return the best delay and the distances
    return best_delay, distances


###################################################################################################
# Functions for Task 2
###################################################################################################
def linear_system(t, y, A):
    """
    function to return vector field of a single point (linear)
    :param t: time (for solve_ivp)
    :param y: single point
    :param A: coefficient matrix, found with least squares
    :return: derivative for point y
    """
    return A @ y


def trajectory(x0, x1, func, args, end_time=0.1, plot=False):
    """
    x0: the data at time 0
    x1: the data at unknown time step after 0
    func: to get derivative for next steps generation
    end_time: end time for the simulation
    plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns points at time end_time
    """
    # the fixed time for system to evaluation
    t_eval = np.linspace(0, end_time, 100)
    sols = []
    x1_pred = []
    for i in range(len(x0)):
        sol = solve_ivp(func, [0, end_time], x0[i], args=args, t_eval=t_eval)  # solve initial value problem for a given point
        x1_pred.append([sol.y[0, -1], sol.y[1, -1]])  # save the final solution

        if plot:
            plt.scatter(x1[i, 0], x1[i, 1], c='red',s= 6)
            plt.scatter(sol.y[0, :], sol.y[1, :], c='limegreen',s=3)
    if plot:
        plt.rcParams["figure.figsize"] = (12,10)
        plt.show()
    return x1_pred


def create_phase_portrait_matrix(A: np.ndarray, title_suffix: str, save_plots=False,
                                 save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax
    :param A: system's (2x2 matrix in our case)
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 10  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=1.0)
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def estimate_vector(x0, x1, delta_t):
    """Estimate the field vector.
    :param x0: is the first input position
    :param x1: is the delta time later position
    :param delta_t: the time step
    :return:
    """
    return (x1 - x0) / delta_t
