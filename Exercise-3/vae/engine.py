import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal
from model_utils import save_best_model


#Function to create the distributions in device
def make_distribution_in_device(mu, sigma, device):
    mu = mu.to(device)
    sigma = sigma.to(device)
    return MultivariateNormal(mu, sigma)



def train_step(model, dataloader, opt, beta, device, clip_grad_max_norm=1.0):
    model.train()  # Set the model to training mode
    total_loss = 0
    kl_loss_total = 0
    total_likelihood = 0

    for data in dataloader:

        #Process the data differently if it is the MNIST dataset or the FireEvac dataset
        if isinstance(data, list):
            X, y = data

        else:
            X = data

        # Move batch image to device
        batch = X.to(device)
        

        # Zero the gradients
        opt.zero_grad()

        # Forward pass
        output_dist, posterior = model(batch)

        #Likelihood
        likelihood = output_dist.log_prob(batch.view(batch.size(0), -1)).mean()
       
        posterior_dist =  make_distribution_in_device(posterior[0], posterior[1], device=device)

        prior_mean = model.prior[0].expand(batch.size(0), -1)  
        prior_covariance = model.prior[1].expand(batch.size(0), -1, -1)  

        prior_dist = make_distribution_in_device(prior_mean, prior_covariance, device=device)

        #KL Divergence
        kl_div = kl_divergence(posterior_dist, prior_dist).mean()
        

        # Calculate the ELBO (Evidence Lower Bound)
        elbo = likelihood - beta * kl_div

        # Calculate the negative ELBO
        loss = -elbo

        # Backward pass
        loss.backward()

        torch_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)

        # Update weights
        opt.step()

        # Accumulate total loss for this epoch
        total_loss += loss.item()
        kl_loss_total += kl_div.item()
        total_likelihood += likelihood.item()

    # Calculate average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_kl_loss = kl_loss_total / len(dataloader.dataset)
    avg_likelihood = total_likelihood / len(dataloader.dataset)

    return avg_loss, avg_kl_loss, avg_likelihood


def test_step(model, dataloader, beta, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    kl_loss_total = 0
    total_likelihood = 0

    with torch.no_grad():
        for data in dataloader:
            
            if isinstance(data, list):
                X, y = data

            else:
                X = data

            # Move batch image to device
            batch = X.to(device)

            # Forward pass
            output_dist, posterior = model(batch)

            likelihood = output_dist.log_prob(batch.view(batch.size(0), -1)).mean()
       
            posterior_dist =  make_distribution_in_device(posterior[0], posterior[1], device=device)

            prior_mean = model.prior[0].expand(batch.size(0), -1)  
            prior_covariance = model.prior[1].expand(batch.size(0), -1, -1)  
        
            prior_dist = make_distribution_in_device(prior_mean, prior_covariance, device=device)

            kl_div = kl_divergence(posterior_dist, prior_dist).mean()

            # Calculate the ELBO (Evidence Lower Bound)
            elbo = likelihood - beta * kl_div

            loss = -elbo

            # Accumulate total loss for this epoch
            total_loss += loss.item()
            kl_loss_total += kl_div.item()
            total_likelihood += likelihood.item()

    # Calculate average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_kl_loss = kl_loss_total / len(dataloader.dataset)
    avg_likelihood = total_likelihood / len(dataloader.dataset)

    return avg_loss, avg_kl_loss, avg_likelihood

#Train the model for a given number of epochs
def train(model, train_dataloader, test_dataloader, optimizer, beta, epochs, device, save_path):
    results = {
        "train_loss": [],
        "train_kl_loss": [],
        "test_loss": [],
        "test_kl_loss": [],
        "train_likelihood": [],
        "test_likelihood": []
    }

    model.to(device)  # Move the model to the specified device
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"---------------Epoch {epoch + 1}/{epochs}:-------------------")

        # Training
        total_loss, kl_loss, likelihood = train_step(model, train_dataloader, optimizer, beta, device)
        results["train_loss"].append(total_loss)
        results["train_kl_loss"].append(kl_loss)
        results["train_likelihood"].append(likelihood)
        print(f"Training Loss: {total_loss}, Training Likelihood: {likelihood}, Training KL Loss: {kl_loss}")

        # Testing
        test_total_loss, test_kl_loss, test_likelihood = test_step(model, test_dataloader, beta, device)
        results["test_loss"].append(test_total_loss)
        results["test_kl_loss"].append(test_kl_loss)
        results["test_likelihood"].append(test_likelihood)
        print(f"Testing Loss: {test_total_loss}, Testing Likelihood: {test_likelihood}, Testing KL Loss: {test_kl_loss}")

        best_val_loss = save_best_model(model, test_total_loss, best_val_loss, save_path)

    return results



