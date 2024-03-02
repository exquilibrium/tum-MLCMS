import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait(A, X, Y):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x');
    ax0.set_aspect(1)
    return ax0

def plot_phase_portrait_in_axis(A, X, Y, axis):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)


    axis.streamplot(X, Y, U, V, density=[0.5, 1])
    axis.set_title('Streamplot for linear vector field A*x');
    axis.set_aspect(1)

import scipy

def plot_arbitrary_phase_portrait_in_axis(f, X, Y, axis, rotate = 0):
    Xrv = np.ravel(X)
    Yrv = np.ravel(Y)
    n = Xrv.shape[0]
    UV = np.zeros((2,n))

    for i in range(n):
        UV[:,i] = f(np.array([Xrv[i], Yrv[i]]))
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)


    axis.streamplot(X, Y, U, V, density=[3, 3])
    axis.set_aspect(1)


if __name__ == "__main__":
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    fig = plt.figure()
    gs = fig.add_gridspec(3, 4, right=3, top=3, hspace=1)
    ax1 = fig.add_subplot(gs[0, 1])
    def f(x):
        return np.array([
        1*x[0] - x[1] - x[0] * (x[0]*x[0] + x[1]*x[1]),
        x[0] + 1*x[1] - x[1] * (x[0]*x[0] + x[1]*x[1])])


    plot_arbitrary_phase_portrait_in_axis(f,X,Y,ax1)