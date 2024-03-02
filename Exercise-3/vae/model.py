import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.distributions import MultivariateNormal


#Encoder class
class Encoder(nn.Module):
    def __init__(self, input_size: int, num_hidden_layers: int, hidden_units: int, latent_dims: int):
        """
        Encoder class for a Variational Autoencoder (VAE)

        Args:
            input_size (int): Size of the input data
            num_hidden_layers (int): Number of hidden layers in the encoder
            hidden_units (int): Number of units in each hidden layer
            latent_dims (int): Dimensionality of the latent space
        """
        super(Encoder, self).__init__()


        # Define a list to hold the hidden layers
        self.hidden_layers = nn.ModuleList()

        # Add the first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_units))

        # Add the specified number of hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))

        # Output layers for mean and standard deviation computation
        self.mean_layer = nn.Linear(hidden_units, latent_dims)
        self.stddev_layer = nn.Linear(hidden_units, latent_dims)


    def forward(self, x):
        """
        Forward pass of the encoder

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Mean and standard deviation of posterior distribution q(z|x)
        """
        x = torch.flatten(x, start_dim=1)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Compute mean and standard deviation
        mu = self.mean_layer(x)
        sigma = torch.exp(self.stddev_layer(x))

        #Return mean and standard deviation 
        return mu, sigma
    

#Return the mean and covariance matrix of the prior distribution p(z)
def prior(latent_dims: int):
    mu = torch.zeros(latent_dims)
    stddev = torch.ones(latent_dims)
    return (mu, torch.diag_embed(stddev))


#Decoder class
class Decoder(nn.Module):
    def __init__(self, output_size: int, num_hidden_layers: int, hidden_units: int,  latent_dims: int):
        """
        Decoder class for a Variational Autoencoder (VAE)

        Args:
            output_size (int): Size of the output
            num_hidden_layers (int): Number of hidden layers in the decoder
            hidden_units (int): Number of units in each hidden layer
            latent_dims (int): Dimensionality of the latent space
        """
        super(Decoder, self).__init__()

        # List to store hidden layers
        self.hidden_layers = nn.ModuleList()

        # Add the first hidden layer
        self.hidden_layers.append(nn.Linear(latent_dims, hidden_units))

        # Add the specified number of additional hidden layers
        for _ in range(num_hidden_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))


        self.mean = nn.Linear(hidden_units, output_size)
        #Parameter for the standard deviation
        self.log_stddev = nn.Parameter(torch.randn(1, output_size))

    def forward(self, z):
        """
        Forward pass of the decoder

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Mean of the distribution of the reconstructed output
        """
        x = z

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # assumes that the distribution of the data p(x|z) is a multivariate Gaussian with diagonal covariance matrix
        mean = torch.sigmoid(self.mean(x))

        return mean

    
#VAE class
class VAE(nn.Module):
    def __init__(self, input_size: int, num_hidden_layers: int, hidden_units: int, latent_dims: int):
        """
        Variational Autoencoder (VAE) class

        Args:
            input_size (int): Size of the input data
            num_hidden_layers (int): Number of hidden layers in the encoder and decoder
            hidden_units (int): Number of units in each hidden layer
            latent_dims (int): Dimensionality of the latent space
        """
        super(VAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.prior = prior(latent_dims)
        self.encoder = Encoder(input_size, num_hidden_layers, hidden_units, latent_dims)
        self.decoder = Decoder(input_size, num_hidden_layers, hidden_units, latent_dims)
        
        

    def forward(self, x):
        """
        Forward pass of the VAE

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple: Distribution of the reconstructed output and the posterior distribution's mean and covariance matrix
        """

        mean_e, stddev_e = self.encoder(x)
        covariance_e = torch.diag_embed(stddev_e**2)
       
        # Sample from the posterior distribution q(z|x)
        z = self.sample_posterior(mean_e, covariance_e)

        # Forward pass through the decoder
        mean = self.decoder(z)

        # Compute the standard deviation to be non-negative
        stddev = torch.exp(self.decoder.log_stddev)
        stddev = stddev.view(-1)
       
                
        # Return the distribution of p(x|z) and q(z|x) distribution's mean and covariance matrix
        return MultivariateNormal(mean, (stddev**2).diag()), (mean_e, covariance_e)
    

    #Sample from the posterior distribution q(z|x) a value z
    def sample_posterior(self, mu, covariance):
        """
        Sample from the posterior distribution q(z|x)

        Args:
            mu (torch.Tensor): Mean of the posterior distribution
            covariance (torch.Tensor): Covariance matrix of the posterior distribution

        Returns:
            torch.Tensor: Sampled value from the posterior distribution
        """
        # Sample from the posterior distribution q(z|x)
        z = MultivariateNormal(mu, covariance).rsample()

        return z
        






