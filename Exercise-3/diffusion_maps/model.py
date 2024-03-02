import numpy as np
import scipy as sp


class DiffusionMap:
    def __init__(self, numEigen=1):
        self.numEigen = numEigen
        self.dMap = None
        self.lambdas = None
        self.phis = None
        self.epsilon = None
        self.qMatInvSqr = None
        self.tMat = None

    """
    Helper function for calculating sparse distance matrix only on close neighbours.
    :param data: The data to calculate a distance matrix on.
    :param max_dist: Max distance of a neighbour.
    :return: Returns a sparse distance matrix as ndarry.
    """
    def to_distance_matrix(self, data, max_dist):
        kd_tree = sp.spatial.KDTree(data)
        d_mat = kd_tree.sparse_distance_matrix(kd_tree, max_distance=max_dist)
        d_mat = d_mat.toarray()
        return d_mat

    """
    Helper function for calculating sparse diagonal matrix from sum of rows.
    :param matrix: The matrix to diagonalize by summing the rows.
    """

    def diag_from_row_sum(self, matrix):
        return sp.sparse.diags(matrix.sum(axis=1, dtype='float'), format="csc")

    """
    Fit the model's Diffusion Map components.
    :param data: The data to fit to. Datapoints stored in rows.
    :param max_dist: The maximum distance betweem neighbours for KDTree.
    """

    def construct_t(self, data, max_dist=1.0):
        # 1.) Form distance matrix
        d_mat = self.to_distance_matrix(data, max_dist)

        # 2.) Set diameter of dataset to 5%
        epsilon = 0.05 * np.max(d_mat)

        # 3.) Form un-normalized kernel matrix
        w_mat = np.exp(-np.square(d_mat) / epsilon)

        # 4.) 5.) Form normalized kernel matrix
        p_mat = self.diag_from_row_sum(w_mat)
        p_mat_inv = sp.sparse.linalg.inv(p_mat)
        k_mat = p_mat_inv.dot(sp.sparse.csc_matrix.dot(w_mat, p_mat_inv))

        # 6.) 7.) Form normalized symmetric matrix
        q_mat = self.diag_from_row_sum(k_mat)
        q_mat_inv_sqr = np.sqrt(sp.sparse.linalg.inv(q_mat))
        t_mat = q_mat_inv_sqr.dot(sp.sparse.csc_matrix.dot(k_mat, q_mat_inv_sqr))

        # Save values for later
        self.epsilon = epsilon
        self.qMatInvSqr = q_mat_inv_sqr
        self.tMat = t_mat

    """
    Compute the eigenvalues and eigenvectors. 
    Compute the diffusion map from the eigen-decomposition
    """

    def compute_eigen(self):
        # 8.) Find L+1 largest eigenvalues and eigenvectors
        evals, evecs = sp.sparse.linalg.eigs(self.tMat, k=(self.numEigen+1), which='LR')
        ix = evals.argsort()[::-1][1:]  # filters out phi_0

        # 9.) Calculate eigenvalues LAMBDA
        evals = np.real(evals[ix])
        lambdas = np.sqrt(np.power(evals, 1/self.epsilon))

        # 10.) Calculate eigenvectors PHI
        evecs = np.real(evecs[:, ix])
        phis = self.qMatInvSqr.dot(evecs)

        # Calculate diffusion map
        dMap = np.matmul(phis, np.diag(lambdas))

        # Save values for later
        self.lambdas = lambdas
        self.phis = phis
        self.dMap = dMap

    """
    Fit the model's Diffusion Map components.
    :param data: The data to fit to. Datapoints stored in rows.
    :param max_dist: The maximum distance betweem neighbours for KDTree.
    """

    def fit(self, data, max_dist=1.0):
        # 1.) - 7.) Calculate symmetric matrix T
        self.construct_t(data, max_dist)

        # 8.) - 10.) Calculate Laplace-Beltrami Operator
        self.compute_eigen()

    """
    Calculates the accuracy of the diffusion map
    :param data:
    :param phi:
    """

    def acc(self, data, dmap, lth, max_dist=1000):
        d_mat = self.to_distance_matrix(data, max_dist=max_dist)

        arr = np.array([dmap[: ,0],dmap[: ,lth]]).T
        dmd_mat = self.to_distance_matrix(arr, max_dist=max_dist)

        # Calculate MSE
        return np.mean(d_mat - dmd_mat)

    """
    Calculates the accuracy of the diffusion map
    :param data: Input data
    :param dmap_1: Eigenfunction 1
    :param dmap_2: Eigenfunction 2
    :param max_dist: max_dist for KDTree
    """

    def acc(self, data, dmap_1, dmap_2, max_dist=1000):
        d_mat = self.to_distance_matrix(data, max_dist=max_dist)

        arr = np.array([dmap_1, dmap_2]).T
        dmd_mat = self.to_distance_matrix(arr, max_dist=max_dist)

        # Calculate MSE
        return np.mean(d_mat - dmd_mat)