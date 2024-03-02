import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import json
import matplotlib.pyplot as plt

result = {'exp_name': [],'architecture':[], 'train_loss':[], 'test_loss':[]}


def experiment(datasetC, datasetB, hidden_layers_structure, num_epochs):
    """
    Run a series of experiments with different dataset combinations.

    Args:
        dataset_c (numpy.ndarray): Dataset C.
        dataset_b (numpy.ndarray): Dataset B.

    Returns:
        dict: Experiment results.
    """
    result['exp_name'] = []
    result['architecture'] = []
    result['train_loss'] = []
    result['test_loss'] = []
    
    
    train_and_test("C/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("B/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("C/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("B/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("C+B/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("C+B/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    train_and_test("C+B/C+B", datasetC, datasetB, hidden_layers_structure, num_epochs)

    with open('result.json', 'w') as file:
        json.dump(result, file)

    return result

def experiment_one(datasetC, datasetB, hidden_layers_structure, num_epochs, type_name):
    """
    Run a series of experiments with different dataset combinations.

    Args:
        datasetC (numpy.ndarray): Dataset C.
        datasetB (numpy.ndarray): Dataset B.
        hidden_layers_structure (list): List specifying the architecture of hidden layers.
        num_epochs (int): Number of epochs for training.
        type_name (str): Type of experiment to run.

    Returns:
        dict: Experiment results.
    """
    result['exp_name'] = []
    result['architecture'] = []
    result['train_loss'] = []
    result['test_loss'] = []
    
    if(type_name == "C/C"):
        train_and_test("C/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "B/B"):
        train_and_test("B/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "C/B"):
        train_and_test("C/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "B/C"):
        train_and_test("B/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "C+B/C"):
        train_and_test("C+B/C", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "C+B/B"):
        train_and_test("C+B/B", datasetC, datasetB, hidden_layers_structure, num_epochs)
    if(type_name == "C+B/C+B"):
        train_and_test("C+B/C+B", datasetC, datasetB, hidden_layers_structure, num_epochs)

    # with open('result.json', 'w') as file:
    #     json.dump(result, file)

    return result


def train_and_test(name, dataset_1, dataset_2, hidden_layers_structure = [[2],[3],[4],[4,2],[5,2],[5,3],[6,3],[10,4]], num_epochs = 50):
    """
    Train and test a model with different datasets and architectures.

    Args:
        name (str): Name of the experiment.
        dataset_1 (numpy.ndarray): First dataset.
        dataset_2 (numpy.ndarray, optional): Second dataset.

    Returns:
        None
    """

    # Check if a GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Splitting the dataset into training, validation, and test sets

    if name == "C/C":
        train_data, val_data = train_test_split(dataset_1, test_size=0.2, random_state=15)
        train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=15)
    if name == "B/B":
        train_data, val_data = train_test_split(dataset_2, test_size=0.2, random_state=15)
        train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=15)
    if name == "C/B":
        train_data, val_data = train_test_split(dataset_1, test_size=0.2, random_state=15)
        test_data = dataset_2
    if name == "B/C":
        train_data, val_data = train_test_split(dataset_2, test_size=0.2, random_state=15)
        test_data = dataset_1
    if name == "C+B/C":
        train_data, test_data = train_test_split(dataset_1, test_size=0.3, random_state=15)
        train_data = np.concatenate((train_data, dataset_2), axis=0)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=15)
    if name == "C+B/B":
        train_data, test_data = train_test_split(dataset_2, test_size=0.3, random_state=15)
        train_data = np.concatenate((train_data, dataset_1), axis=0)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=15)
        # train_data, _ = train_test_split(train_data, test_size=0.78, random_state=15)
    if name == "C+B/C+B":
        dataset_3 = np.concatenate((dataset_1, dataset_2), axis=0)
        train_data, val_data = train_test_split(dataset_3, test_size=0.2, random_state=15)
        train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=15)
    
    # Create dataset objects
    train_dataset = MyDataset(train_data)
    val_dataset = MyDataset(val_data)
    test_dataset = MyDataset(test_data)

    # Create DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    for h in hidden_layers_structure:
        print(f"Experiment {name}： {len(result['exp_name'])+1}/{len(hidden_layers_structure)}  ---   Model: {str(h)}")
        model = SimpleModel(h).to(device).float()
        criterion = nn.MSELoss().to(device).float()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        average_train_loss, average_test_loss = train(num_epochs, train_loader, test_loader, val_loader, device, model, criterion, optimizer)
        result['exp_name'].append(name)
        result['architecture'].append(str(h))
        result['train_loss'].append(average_train_loss)
        result['test_loss'].append(average_test_loss)
        
def train(num_epochs, train_loader, test_loader, val_loader, device, model, criterion, optimizer, patience=3):
    """
    Train the model and evaluate its performance on validation and test sets.

    Args:
        num_epochs (int): Number of training epochs.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for training (cuda or cpu).
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim): Optimization algorithm.

    Returns:
        tuple: Average training loss and average test loss.
    """
    best_val_loss = float('inf')
    counter = 0  # Counter for consecutive epochs without improvement
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}: ')
        model.train()

        total_train_loss = 0.0  # Variable to accumulate total training loss
        total_batch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            features, labels = batch['features'], batch['labels']
    #         print(labels)

            # Move data to GPU and ensure consistent data types
            features, labels = features.to(device).float(), labels.to(device).float()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            # Compute the loss
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()  # Accumulate training loss for this batch
            total_batch_loss += loss.item()  # Accumulate training loss for this batch

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            print_every = 1000
            # Print batch-wise loss every n batches
            if (batch_idx + 1) % print_every == 0:
                average_batch_loss = total_batch_loss / print_every
                print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Average Loss: {average_batch_loss}')
                total_batch_loss = 0.0  # Reset total training loss

        # Calculate average training loss for the epoch
        average_train_loss = total_train_loss / len(train_loader)
        print(f'  Average Training Loss: {average_train_loss}')

        # Validate the model after each epoch
        model.eval()
        with torch.no_grad():
            val_losses = []

            for val_batch_idx, val_batch in enumerate(val_loader):
                val_features, val_labels = val_batch['features'], val_batch['labels']

                # Move data to GPU
                val_features, val_labels = val_features.to(device).float(), val_labels.to(device).float()

                # Forward pass
                val_outputs = model(val_features)

                # Compute the validation loss
                val_loss = criterion(val_outputs, val_labels)
                val_losses.append(val_loss.item())  # Accumulate validation loss for this batch

            average_val_loss = np.average(val_losses)

        
        print(f'  Average Training Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            counter = 0  # Reset counter when there is improvement
        else:
            counter += 1

        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}. No improvement for {patience} consecutive epochs.')
            break

    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0  # Variable to accumulate total test loss

        for test_batch_idx, test_batch in enumerate(test_loader):
            test_features, test_labels = test_batch['features'], test_batch['labels']

            # Move data to GPU
            test_features, test_labels = test_features.to(device).float(), test_labels.to(device).float()

            # Forward pass
            test_outputs = model(test_features)

            # Compute the test loss
            test_loss = criterion(test_outputs, test_labels)
            total_test_loss += test_loss.item()  # Accumulate test loss for this batch

        average_test_loss = total_test_loss / len(test_loader)

    print(f'  Average Test Loss: {average_test_loss}')

    return average_train_loss, average_test_loss

def plot_architecture_subplots(result):
    # Extract unique values for 'exp_name' and 'architecture'
    exp_name = ['C/C','B/B','C/B','B/C','B+C/C','B+C/B','B+C/B+C']
    architecture = [[3], [4], [4, 2], [5, 2], [5, 3], [6, 3]]

    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    # Iterate through each architecture value
    for i, arch_value in enumerate(architecture):
        train_losses = []
        test_losses = []

        # Extract relevant data for the current architecture value
        for index, value in enumerate(result['architecture']):
            if value == str(arch_value):
                train_losses.append(result['train_loss'][index])
                test_losses.append(result['test_loss'][index])

        # Plot the data on the corresponding subplot
        row, col = divmod(i, 3)
        axes[row, col].plot(exp_name, train_losses, linewidth=5, label='Train Loss', color=(88/255, 88/255, 162/255))
        axes[row, col].plot(exp_name, test_losses, linewidth=5, label='Test Loss', color=(197/255, 197/255, 197/255))

        # Set title and axis labels for each subplot
        axes[row, col].set_title(str(tuple(eval(arch_value))))
        axes[row, col].set_ylabel('MSE')

    # Adjust layout to ensure proper spacing between subplots
    plt.tight_layout()
    plt.legend()
    # Display the plot
    plt.show()

def plot_experiment_subplots(result):
    # Extract unique values for 'exp_name' and 'architecture'
    exp_name = list(set(result['exp_name']))
    architecture = list(set(result['architecture']))

    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    # Iterate through each architecture value
    for i, exp_value in enumerate(exp_name):
        train_losses = []
        test_losses = []

        # Extract relevant data for the current architecture value
        for index, value in enumerate(result['exp_name']):
            if value == str(exp_value):
                train_losses.append(result['train_loss'][index])
                test_losses.append(result['test_loss'][index])

        # Plot the data on the corresponding subplot
        row, col = divmod(i, 3)
        print(str(architecture))
        axes[row, col].plot(str(architecture), train_losses, linewidth=5, label='Train Loss', color=(88/255, 88/255, 162/255))
        axes[row, col].plot(str(architecture), test_losses, linewidth=5, label='Test Loss', color=(197/255, 197/255, 197/255))

        # Set title and axis labels for each subplot
        axes[row, col].set_title(exp_value)
        axes[row, col].set_ylabel('MSE')

    # Adjust layout to ensure proper spacing between subplots
    plt.tight_layout()
    plt.legend()
    # Display the plot
    plt.show()


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the first 41 columns are features (training data) and the last two columns are labels
        features = self.data[idx, :41]
        labels = self.data[idx, 41:]

        return {'features': features, 'labels': labels}
    
class SimpleModel(nn.Module):
    def __init__(self, hidden_layers, input_size = 41, output_size=1):
        super(SimpleModel, self).__init__()

        # Input layer to the first hidden layer
        self.layers = [nn.Linear(input_size, hidden_layers[0]), nn.Sigmoid()]

        # Add additional hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.Sigmoid()])

        # Last hidden layer to the output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Put all layers into Sequential
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
    
