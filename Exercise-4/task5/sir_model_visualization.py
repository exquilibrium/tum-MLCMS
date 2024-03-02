import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from sir_model import *

# Simulation parameters
t_0 = 0
t_end = 1000
NT = t_end - t_0
# If these error tolerances are set too high, the solution will be qualitatively (!) wrong.
rtol = 1e-8
atol = 1e-8
sim_params = (t_0, t_end, NT, rtol, atol)

# SIR model parameters
A = 20  # "recruitment rate" (or birth rate) of susceptible population
d = 0.1  # per capita natural death rate
nu = 1  # per capita disease-induced death rate
mu0 = 10  # minimum recovery rate
mu1 = 10.45  # maximum recovery rate
beta = 11.5  # average number of adequate contacts per unit time with infectious individuals
b = 0.022  # 0.01 # try to set this to 0.01, 0.020, ..., 0.022, ..., 0.03
sir_params = (A, d, nu, mu0, mu1, beta, b)

# Default initial SIR values
random_state = 12345
rng = np.random.default_rng(random_state)
simInit = rng.uniform(low=(190, 0, 1), high=(199, 0.1, 8), size=(3,))

# Plot simulation function
def plotSimulation(init=simInit, simPar=sim_params, sirPar=sir_params, info=True, new_b=b):
    """
    Plots of the SIR simulation

    Parameters:
        init (list): initial SIR values
        simPar (tuple): simulation parameters
        sirPar (tuple): SIR parameters
        info (bool): print reproduction number R0 and asymptotic stability
        new_b (float): new b
    """
    # Extract parameters
    SIM0 = init
    t_0, t_end, NT, rtol, atol = simPar
    A, d, nu, mu0, mu1, beta, b = sirPar
    b = new_b

    # Information
    if info:
        print("Reproduction number R0=", R0(beta, d, nu, mu1))
        print('Globally asymptotically stable if beta <=d+nu+mu0. This is', beta <= d + nu + mu0)

    # Simulation
    time = np.linspace(t_0, t_end, NT)
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b),
                    method='LSODA', rtol=rtol, atol=atol)

    # Plot "S","I","R"
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label='1E0*susceptible')
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label='1E3*infective')
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label='1E1*removed')
    ax[0].set_xlim([0, 500])
    ax[0].legend()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    # Plot "Recovery rate", "R"
    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2 * sol.y[1], label='1E2*infective')
    ax[1].set_xlim([0, 500])
    ax[1].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    # Plot "Indicator function"
    I_h = np.linspace(-0., 0.05, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b))
    ax[2].plot(I_h, 0 * I_h, 'r:')
    # ax[2].set_ylim([-0.1,0.05]) # rescale y-axis
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")
    fig.tight_layout()


def plotTrajectory(init=simInit, simPar=sim_params, sirPar=sir_params, lines=False, new_b = b, opacity=0.3):
    """
    Plot the SIR trajectory

    Parameters:
        init (list): initial SIR values
        simPar (tuple): simulation parameters
        sirPar (tuple): SIR parameters
        lines (bool): draw lines into scatter plot
        new_b (float): new b
        opacity (float): opacity of plot
    """
    # Extract parameters
    SIM0 = [init[0], init[1], init[2]]
    t_0, t_end, NT, rtol, atol = simPar
    A, d, nu, mu0, mu1, beta, b = sirPar
    b = new_b

    # Plot trajectory
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    time = np.linspace(t_0, 15000, NT)

    # color gradients map
    cmap = ["BuPu", "Purples", "bwr"][2]

    # what happens with this initial condition when b=0.022?
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b),
                    method='DOP853', rtol=rtol, atol=atol)
    if lines:
        ax.plot(sol.y[0], sol.y[1], sol.y[2], 'r-', alpha=opacity)  # connect dots
    ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap)

    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")

    ax.set_title("SIR trajectory")
    fig.tight_layout()

def plotTrajectoryXY(init=simInit, simPar=sim_params, sirPar=sir_params, lines=False, new_b = b, opacity=0.3, ax0=0, ax1=1):
    """
    Plot the SIR trajectory in XY plane

    Parameters:
        init (list): initial SIR values
        simPar (tuple): simulation parameters
        sirPar (tuple): SIR parameters
        lines (bool): draw lines into scatter plot
    """
    # Extract parameters
    SIM0 = [init[0], init[1], init[2]]
    t_0, t_end, NT, rtol, atol = simPar
    A, d, nu, mu0, mu1, beta, b = sirPar
    b = new_b

    # Plot trajectory
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    time = np.linspace(t_0, 15000, NT)

    # color gradients map
    cmap = ["BuPu", "Purples", "bwr"][2]

    # what happens with this initial condition when b=0.022?
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b),
                    method='DOP853', rtol=rtol, atol=atol)
    if lines:
        ax.plot(sol.y[ax0], sol.y[ax1], 'r-', alpha=opacity)  # connect dots
    ax.scatter(sol.y[ax0], sol.y[ax1], s=1, c=time, cmap=cmap)

    ax.set_xlabel("S")
    ax.set_ylabel("I")

    ax.set_title("SI trajectory")
    fig.tight_layout()

def plotBifurcation(init=simInit, simPar=sim_params, sirPar=sir_params, new_b=b):
    # Extract parameters
    """
    Plot the R0-I bifurcation over time new_t

    Parameters:
        init (list): initial SIR values
        simPar (tuple): simulation parameters
        sirPar (tuple): SIR parameters
        lines (bool): draw lines into scatter plot
        new_b (float): new b
        opacity (float): opacity of plot
    """


    SIM0 = [init[0], init[1], init[2]]
    t_0, t_end, NT, rtol, atol = simPar
    A, d, nu, mu0, mu1, beta, b = sirPar
    b = new_b

    time = np.linspace(t_0, 15000, NT)

    # Lists
    mus = [mu0 + (mu1 - mu0) * i / 600 for i in range(1000)]
    r0s = [R0(beta, d, nu, m) for m in mus]
    listI = []

    for mu in mus:
        sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu, beta, A, d, nu, b),
                        method='DOP853', rtol=rtol, atol=atol)
        listI.append(sol.y[1]) # sol.y[1] => number of Infected listed by time

    # Plot trajectory
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    time = np.linspace(t_0, t_end, NT)

    # color gradients map
    cmap = ["BuPu", "Purples", "bwr"][2]

    for i in range(100):
        parI = [row[i * 10] for row in listI]
        parT = [i for _ in range(1000)]
        ax.scatter(r0s, parT, parI, s=1, c=time, cmap=cmap, alpha=0.05)

    ax.set_xlabel("R0")
    ax.set_ylabel("t*10")
    ax.set_zlabel("I")

    ax.set_title("R0-t-I trajectory")
    fig.tight_layout()

def plotBifurcationT(init=simInit, simPar=sim_params, sirPar=sir_params, new_b=b, new_t=0):
    # Extract parameters
    """
    Plot the R0-I bifurcation at time new_t

    Parameters:
        init (list): initial SIR values
        simPar (tuple): simulation parameters
        sirPar (tuple): SIR parameters
        lines (bool): draw lines into scatter plot
        new_b (float): new b
        new_t (float): time slice to plot
    """
    SIM0 = [init[0], init[1], init[2]]
    t_0, t_end, NT, rtol, atol = simPar
    A, d, nu, mu0, mu1, beta, b = sirPar
    b = new_b

    time = np.linspace(t_0, 15000, NT)

    # Lists
    mus = [mu0 + (mu1 - mu0) * i / 600 for i in range(1000)]
    r0s = [R0(beta, d, nu, m) for m in mus]
    listI = []

    for mu in mus:
        sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu, beta, A, d, nu, b),
                        method='DOP853', rtol=rtol, atol=atol)
        listI.append(sol.y[1][new_t])  # sol.y[1] => number of Infected listed by time

    # Plot trajectory
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    time = np.linspace(t_0, t_end, NT)

    ax.plot(r0s, listI, alpha=0.9)

    ax.set_xlabel("R0")
    ax.set_ylabel("I")

    ax.set_title("R0-I trajectory")
    fig.tight_layout()