import numpy as np
import pca.helpers as helpers


class ReversiblePCA:
    def __init__(self):
        self._principal_components = None
        self._s = None
        self._u = None
        self._mean = None

    @property
    def principal_components(self):
        return self._principal_components

    @property
    def eigen_values(self):
        return self._s * self._s

    """
    Fit the model's PCA components so you can call the transform function later.
    :param data: The data to fit to. Datapoints stored in rows.
    """

    def fit(self, data, use_covariance = True):
        self._mean = helpers.mean(data)
        data_centered = (data - self._mean)
        u, s, vh = None, None, None
        if use_covariance:
            u, s, vh = np.linalg.svd(np.cov(data_centered.T))
        else:
            u, s, vh = np.linalg.svd(data_centered)
        self._principal_components = vh
        self._s = s

    """
    Fit the model to the data and return the transformed data.
    :param data: The data to fit to and transform. Datapoints stored in rows.
    """

    def fit_transform(self, data, number_of_components):
        self.fit(data)
        return self.transform(data, number_of_components)

    """
    Transform the data based on the PCA components learned in the fit function.
    :param data: The data to transform. Datapoints stored in rows.
    :param number_of_components: The dimensionality of the resulting space. 
    Uses the given number of most principal components.
    :param shared_mean: Whether to use the mean of this data.
    """

    def transform(self, data, number_of_components=-1, shared_mean=False):
        mean = helpers.mean(data)
        if number_of_components <= 0:
            if shared_mean:
                return self.transform_static(data, self._principal_components, self._mean)
            return self.transform_static(data, self._principal_components, mean)
        diag = self.diagonal(data.shape[1],number_of_components)
        if shared_mean:
            return self.transform_static(data, diag@self._principal_components@diag, self._mean)
        return self.transform_static(data, diag@self._principal_components, mean)

    def diagonal(self,shape,number_of_components):
        diag = np.ones(shape)
        diag[number_of_components:] = 0
        return np.diag(diag)


    """
    Transform the PCA transformed data with the reverse of the fitted transform.
    :param data: The data to reverse transform. Datapoints stored in rows.
    """

    def reverse_transform(self, data, number_of_components = -1):
        if number_of_components < 0:
            number_of_components = data.shape[1]
        diag = self.diagonal(data.shape[1],number_of_components)
        return self.reconstruct(data, diag @ self._principal_components, self._mean)

    """
    Reverse transform the PCA transformed data based on the given PCA components.
    :param data: The data to reverse transform. Datapoints stored in rows.
    :param principal_components: The principal_components stored in rows.
    """

    @staticmethod
    def reconstruct(data, principal_components, mean):
        return (data @ principal_components) + mean

    """
    Transform the data based on the given PCA components.
    :param data: The data to transform. Datapoints stored in rows.
    :param principal_components: The principal_components stored in rows.
    """

    @staticmethod
    def transform_static(data, principal_components, mean):
        return (data - mean) @ principal_components.T
