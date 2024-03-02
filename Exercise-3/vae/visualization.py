import matplotlib.pyplot as plt
import torch
import numpy as np
from umap import UMAP #install umap-learn
from mpl_toolkits.mplot3d import Axes3D





#Compare train and testing loss
def plot_loss(train_loss, test_loss, path):
    """Plots training and testing loss curves.

    Args:
        train_loss (list): List of training loss values.
        test_loss (list): List of testing loss values.
    """
    plt.plot(train_loss, label="Training loss")
    plt.plot(test_loss, label="Testing loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #Save the figure
    plt.savefig(f"{path}loss_comparison.png")


#Compare likelihood and kl loss
def plot_likelihood_and_kl_loss(train_likelihood, test_likelihood, train_kl_loss, test_kl_loss, path):
    """Plots training and testing likelihood and KL loss curves.

    Args:
        train_likelihood (list): List of training likelihood values.
        test_likelihood (list): List of testing likelihood values.
        train_kl_loss (list): List of training KL loss values.
        test_kl_loss (list): List of testing KL loss values.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    axes[0].plot(train_likelihood, label="Training likelihood")
    axes[0].plot(test_likelihood, label="Testing likelihood")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Likelihood")
    axes[0].legend()

    axes[1].plot(train_kl_loss, label="Training KL loss")
    axes[1].plot(test_kl_loss, label="Testing KL loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL loss")
    axes[1].legend()

    fig.tight_layout()
    #Save the figure
    plt.savefig(f"{path}likelihood_and_kl_loss.png")


#Plot reconstructions
def plot_reconstructions(model, test_dataloader, device, path, n=15):
    """Plots n original images and their reconstructions from the VAE.

    Args:
        model (VAE): Trained VAE model.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device on which to perform training and testing (e.g., "cuda" or "cpu").
        n (int, optional): Number of images to plot. Defaults to 15.
    """
    # Set model to evaluation mode
    model.eval()

    # Get a batch of test data
    test_data = next(iter(test_dataloader))[0].to(device)
    model.to(device)

    # Generate reconstructions
    with torch.no_grad():
        distribution, _ = model(test_data)
        means = distribution.mean
        
    # Plot n original images and their reconstructions
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(n, 2), sharex=True, sharey=True)
    for i in range(n):
        axes[0, i].imshow(test_data[i].view(28, 28).cpu(), cmap="gray")
        axes[1, i].imshow(means[i].view(28, 28).cpu(), cmap="gray")

    for ax in axes.flatten():
        ax.axis("off")

    fig.tight_layout()
    #Save the figure
    plt.savefig(f"{path}reconstruction.png")


#Plot generated samples
def plot_generated_samples(model, device, path, n=15):
    """Plots n images generated from the VAE.

    Args:
        model (VAE): Trained VAE model.
        device (torch.device): Device on which to perform training and testing (e.g., "cuda" or "cpu").
        n (int, optional): Number of images to plot. Defaults to 15.
    """
    # Set model to evaluation mode
    model.eval()

    # Generate samples
    with torch.no_grad():
        latent_samples = torch.randn(n, model.latent_dims).to(device)
        generated_samples = model.decoder(latent_samples).view(n, 28, 28).cpu()

    # Plot generated samples
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n, 1), sharex=True, sharey=True)
    for i in range(n):
        axes[i].imshow(generated_samples[i], cmap="gray")

    for ax in axes.flatten():
        ax.axis("off")

    fig.tight_layout()
    #Save the figure
    plt.savefig(f"{path}generated_samples.png")


#Plot latent space
def plot_latent(model, test_dataloader, device, path):
    """
    Plot the latent space of a VAE model

    Args:
        model (nn.Module): VAE model with an encoder.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to which the model and data should be moved.
        path (str): Path to save the plot.
    """
    # Set the model to evaluation mode
    model.eval()

    # Lists to store latent space values and corresponding labels for each digit
    latent_spaces_by_digit = {}
    labels_by_digit = {}

    with torch.no_grad():
        for digit in range(10):
            # Filter data for the current digit
            digit_data = [(inputs, label) for inputs, label in test_dataloader.dataset if label == digit]

            if not digit_data:
                continue # Skip if there is no data for the current digit

            digit_inputs, digit_labels = zip(*digit_data)
            digit_inputs = torch.stack(digit_inputs).to(device)

            # Forward pass through the encoder
            mu, _ = model.encoder(digit_inputs)

            latent_space = mu.cpu().numpy()

            # Store the latent space and labels for the current digit
            latent_spaces_by_digit[digit] = latent_space
            labels_by_digit[digit] = np.array(digit_labels)

    # Scatter plot of the latent space with each digit in a different color
    plt.figure(figsize=(10, 8))
    for digit, latent_space in latent_spaces_by_digit.items():
        plt.scatter(latent_space[:, 0], latent_space[:, 1], label=f'Number {digit}', alpha=0.7)


    plt.title('Latent Space Visualization by Digit')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(f"{path}latent_space.png")
    plt.show()




def plot_latent_umap(model, test_dataloader, device, path):
    """
    Plot the latent space of a VAE model using UMAP for dimensionality reduction.

    Args:
        model (nn.Module): VAE model with an encoder.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to which the model and data should be moved.
        path (str): Path to save the plot.
    """
    # Set the model to evaluation mode
    model.eval()

    # Lists to store latent space values and corresponding labels for each digit
    latent_spaces_by_digit = {}
    labels_by_digit = {}

    with torch.no_grad():
        for digit in range(10):
            # Filter data for the current digit
            digit_data = [(inputs, label) for inputs, label in test_dataloader.dataset if label == digit]

            if not digit_data:
                continue  # Skip if there is no data for the current digit

            digit_inputs, digit_labels = zip(*digit_data)
            digit_inputs = torch.stack(digit_inputs).to(device)

            # Forward pass through the encoder
            mu, _ = model.encoder(digit_inputs)

            latent_space = mu.cpu().numpy()

            # Apply UMAP for dimensionality reduction
            reducer = UMAP(n_components=2)
            latent_space_2d = reducer.fit_transform(latent_space)

            # Store the 2D latent space and labels for the current digit
            latent_spaces_by_digit[digit] = latent_space_2d
            labels_by_digit[digit] = np.array(digit_labels)

    # Scatter plot of the 2D latent space with each digit in a different color
    plt.figure(figsize=(10, 8))
    for digit, latent_space_2d in latent_spaces_by_digit.items():
        plt.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], label=f'Number {digit}', alpha=0.7)

    plt.title('2D Latent Space Visualization by Digit (UMAP)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(f"{path}latent_space_umap.png")
    plt.show()




def plot_latent_3d(model, test_dataloader, device, path):
    """
    Plot the latent space of a VAE model using UMAP for 3D dimensionality reduction.

    Args:
        model (nn.Module): VAE model with an encoder.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to which the model and data should be moved.
        path (str): Path to save the plot.
    """
    # Set the model to evaluation mode
    model.eval()

    # Lists to store latent space values and corresponding labels for each digit
    latent_spaces_by_digit = {}
    labels_by_digit = {}

    with torch.no_grad():
        for digit in range(10):
            # Filter data for the current digit
            digit_data = [(inputs, label) for inputs, label in test_dataloader.dataset if label == digit]

            if not digit_data:
                continue  # Skip if there is no data for the current digit

            digit_inputs, digit_labels = zip(*digit_data)
            digit_inputs = torch.stack(digit_inputs).to(device)

            # Forward pass through the encoder
            mu, _ = model.encoder(digit_inputs)

            latent_space = mu.cpu().numpy()

            # Apply UMAP for 3D dimensionality reduction
            reducer = UMAP(n_components=3)
            latent_space_3d = reducer.fit_transform(latent_space)

            # Store the 3D latent space and labels for the current digit
            latent_spaces_by_digit[digit] = latent_space_3d
            labels_by_digit[digit] = np.array(digit_labels)

    # Scatter plot of the 3D latent space with each digit in a different color
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for digit, latent_space_3d in latent_spaces_by_digit.items():
        ax.scatter(latent_space_3d[:, 0], latent_space_3d[:, 1], latent_space_3d[:, 2], label=f'Number {digit}', alpha=0.7)

    ax.set_title('3D Latent Space Visualization by Digit (UMAP)')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.legend()
    plt.savefig(f"{path}latent_space_umap_3d.png")
    plt.show()





#Plot latent space for the evacuation dataset
def plot_latent_fire_evacuation(model, test_data, path):
    # Set the model to evaluation mode
    model.eval()

    # Encode the data to get the latent representations
    latent_representation = model.encoder(torch.tensor(test_data))[0].detach().numpy()

    # Scatter plot in the latent space with different colors for each class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_representation[:, 0], latent_representation[:, 1], cmap='viridis')
    plt.title('Latent Space of VAE')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig(f"{path}latent_space_fire_evacuation.png")
    plt.show()









    
