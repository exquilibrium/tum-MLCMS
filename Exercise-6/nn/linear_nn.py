from torch import nn
import torch
import os
import sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

class LinearNN(nn.Module):
    """
    A class that represents the linear neural network.

    Attributes:
    model (nn.Sequential): The sequential model of layers in the neural network.
    last (nn.Linear): The last layer of the neural network.
    """

    def __init__(self, hidden_sizes, input_size, output_size, activation_modulef):
        """
        The constructor for LinearNN class.

        Parameters:
        hidden_sizes (tuple): The sizes of the hidden layers.
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        activation_modulef (function): The activation function to use in the layers.
        """
        super().__init__()
        layer_sizes = (input_size,) + tuple(hidden_sizes)
        # Create the model as a sequence of linear layers with activation functions
        self.model = nn.Sequential(
            *[nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation_modulef()) for
              i in range(len(layer_sizes) - 1)])
        self.last = nn.Linear(hidden_sizes[-1], output_size)

    def getlatent(self, x):
        """
        Function to get the latent representation of the input.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The latent representation of the input.
        """
        # Pass the input through all but the last layer of the model
        for i in range(len(self.model)-1):
            x = self.model[i](x)

        # Pass the result through the first part of the last layer
        return self.model[-1][0](x)

    def forward(self, x):
        """
        Function to perform a forward pass through the model.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output of the model.
        """
        return self.last(self.model(x))

class LinearNNWeidmanLike(nn.Module):
    """
    A class that represents a Weidman-like linear neural network.

    Attributes:
    model (nn.Sequential): The sequential model of layers in the neural network.
    t, l, v, sk (nn.Linear): parameters of the model.
    """

    def __init__(self, hidden_sizes, input_size, output_size, activation_modulef):
        """
        The constructor for LinearNNWeidmanLike class.

        Parameters:
        hidden_sizes (tuple): The sizes of the hidden layers.
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        activation_modulef (function): The activation function to use in the layers.
        """
        super().__init__()
        layer_sizes = (input_size,) + tuple(hidden_sizes)
        # Create the model as a sequence of linear layers with activation functions
        self.model = nn.Sequential(
            *[nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation_modulef()) for
              i in range(len(layer_sizes) - 1)])

        self.t = nn.Linear(hidden_sizes[-1], 1)
        self.l = nn.Linear(hidden_sizes[-1], 1)
        self.v = nn.Linear(hidden_sizes[-1], 1)
        self.sk = nn.Linear(hidden_sizes[-1], 1)

    def getlatent(self, x):
        """
        Function to get the latent representation of the input.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The latent representation of the input.
        """
        # Pass the input through all but the last layer of the model
        for i in range(len(self.model)-1):
            x = self.model[i](x)
        # Pass the result through the first part of the last layer
        return self.model[-1][0](x)

    def forward(self, x):
        """
        Function to perform a forward pass through the model.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output of the model.
        """
        # Pass the input through the model
        x = self.model(x)
        t = self.t(x)
        l = self.l(x)
        v = self.v(x)
        sk = self.sk(x)

        # Return the final output
        return v * torch.exp((l-sk) / torch.clamp(v*t,1e-2,5))

class LinearAE(nn.Module):
    """
    A class that represents a linear autoencoder.

    Attributes:
    usemean (bool): Whether to use mean in the forward pass.
    input_size (int): The size of the input layer.
    model (nn.Sequential): The sequential model of layers in the neural network.
    inputbn (nn.BatchNorm1d): Batch normalization for the input layer.
    last_layer (nn.Sequential): The last layer of the neural network.
    """

    def __init__(self, latents, input_size, output_size, activation_modulef, device):
        """
        The constructor for LinearAE class.

        Parameters:
        latents (int): The number of latent variables.
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        activation_modulef (function): The activation function to use in the layers.
        device (str): The device to use for computations.
        """
        super().__init__()
        self.usemean = False
        self.input_size = input_size
        modelouts = latents*latents + (latents if self.usemean else 0)
        layer_sizes = (input_size, modelouts)
        self.latents = latents
        self.device = device
        # Create the model as a sequence of linear layers
        self.model = nn.Sequential(
            *[nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1])) for
              i in range(len(layer_sizes) - 1)])
        # Batch normalization for the input layer
        self.inputbn = nn.BatchNorm1d(input_size)

        self.last_layer = nn.Sequential(nn.BatchNorm1d(latents),nn.Linear(latents, output_size))

        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))

    def forward(self, x):
        """
        Function to perform a forward pass through the model.

        Parameters:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output of the model.
        """
        bs = x.shape[0]
        x = self.model(x)
        # Broadcast the identity matrix to the batch size
        identity = torch.broadcast_to(self.identity.unsqueeze(0),(bs, self.identity.shape[0], self.identity.shape[1]))
        epsilon = torch.randn([bs, self.latents, 1], device = self.device)
        # Calculate the sigma
        sigma = x[:, :self.latents*self.latents].view(bs,self.latents, self.latents)+identity
        if self.usemean:
            # If usemean is True, calculate the mean and add it to the output
            mean = x[:, self.latents*self.latents:]
            return self.last_layer(mean+torch.bmm(sigma, epsilon).view(bs,-1))
        else:
            # If usemean is False, just return the output
            return self.last_layer(torch.bmm(sigma, epsilon).view(bs,-1))