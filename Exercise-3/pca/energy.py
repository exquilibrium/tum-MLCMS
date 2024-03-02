import numpy as np
def first_captures_percentage_energy(eigen_values,percentage):
    return np.min(np.where(relative_cumsums(eigen_values) > percentage))+1

def relative_cumsums(eigen_values):
    return np.cumsum(eigen_values) / np.sum(eigen_values)

def delta(values):
    return values[1:] - values[:-1]

def first_losses_less_than(eigen_values,percentage_loss):
    explained_var_cumsum = relative_cumsums(eigen_values)
    return np.min(np.where((1-explained_var_cumsum) < percentage_loss))+1