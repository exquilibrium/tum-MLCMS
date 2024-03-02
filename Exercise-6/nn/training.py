import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import time
from matplotlib import pyplot as plt
def create_tqdm_bar(iterable, desc):
    """
    Function to create a progress bar for an iterable.

    Parameters:
    iterable (iterable): The iterable to create a progress bar for.
    desc (str): The description of the progress bar.

    Returns:
    tqdm: The progress bar.
    """
    return tqdm(iterable, total=len(iterable), ncols=150, desc=desc)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data, self.targets = data, targets

    def makedataloader(self, batches = 1):
        """
        Makes a rudimentary no shuffle data loader that is entirely in RAM.
        :param batches: num of batches to split into
        :return: dataloader
        """
        if batches == 1:
            return [(torch.tensor(self.data, dtype=torch.float32), torch.tensor(self.targets,
                                                                               dtype=torch.float32))]
        elif batches > self.data.shape[0]:
            return [(torch.tensor(self.data[batch:batch+1], dtype=torch.float32),
                     torch.tensor(self.targets[batch:batch+1],dtype=torch.float32))
                    for batch in range(self.data.shape[0])]
        else:
            perb = len(self)//batches
            return [(torch.tensor(self.data[perb*batch:perb*(batch+1)], dtype=torch.float32),
                     torch.tensor(self.targets[perb*batch:perb*(batch+1)],dtype=torch.float32))
                    for batch in range(batches)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx],
                                                                               dtype=torch.float32)

def validate_model(model, dataloader, loss, device):
    """
    Function to validate a model.

    Parameters:
    model (nn.Module): The model to validate.
    dataloader (DataLoader): The dataloader for the validation data.
    loss (function): The loss function to use.
    device (str): The device to use for computations.

    Returns:
    float, list: The mean of the test scores and the list of test losses.
    """
    test_scores = []
    test_losses = []
    model.eval()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.squeeze().to(device)
        with torch.no_grad():
            outputs = model.forward(inputs).squeeze()
        # If the loss function is MSELoss, calculate the loss between outputs and targets
        if type(loss) is nn.MSELoss:
            l = loss(outputs.cpu(), targets.cpu())
        else:
            # Otherwise, calculate the loss between outputs, targets, and inputs
            l = loss(outputs.cpu(), targets.cpu(), inputs.cpu())
        test_losses.append(l.item())

    return test_losses


class KFACargs:
    """
    A class that represents the arguments for K-FAC.

    Attributes:
    momentum (float): The momentum for K-FAC.
    cov_ema_decay (float): The decay rate for the covariance moving average.
    damping (float): The damping for K-FAC.
    stab_coeff (float): The stability coefficient for K-FAC.
    use_cholesky (bool): Whether to use Cholesky decomposition in K-FAC.
    adjust_momentum (bool): Whether to adjust the momentum in K-FAC.
    acc_iters (int): The number of accumulation iterations in K-FAC.
    """

    def __init__(self, momentum, cov_ema_decay, damping, stab_coeff, use_cholesky, adjust_momentum,
                 acc_iters):
        """
        The constructor for KFACargs class.

        Parameters:
        momentum (float): The momentum for K-FAC.
        cov_ema_decay (float): The decay rate for the covariance moving average.
        damping (float): The damping for K-FAC.
        stab_coeff (float): The stability coefficient for K-FAC.
        use_cholesky (bool): Whether to use Cholesky decomposition in K-FAC.
        adjust_momentum (bool): Whether to adjust the momentum in K-FAC.
        acc_iters (int): The number of accumulation iterations in K-FAC.
        """
        (self.momentum, self.cov_ema_decay,
         self.damping, self.stab_coeff,
         self.use_cholesky,
         self.adjust_momentum, self.acc_iters) = (momentum, cov_ema_decay,
                                                  damping, stab_coeff, use_cholesky,
                                                  adjust_momentum, acc_iters)

    def __str__(self):
        """
        Function to convert the KFACargs object to a string.

        Returns:
        str: The string representation of the KFACargs object.
        """
        return str((self.momentum, self.cov_ema_decay,
         self.damping, self.stab_coeff,
         self.use_cholesky,
         self.adjust_momentum, self.acc_iters))

# Default arguments for K-FAC
defaultKFACargs = KFACargs(0.9, 0.99, 0.03, 16.0,
                           True, True, 2)


def train(model, epochs, validate_every, train_loader, val_loader, lr, lr_decay, device, outp="./nn/models",
          silent=True, usefirstorder=False, kfacargs = defaultKFACargs, extrareg = None, wdecay = 0.2):
    """
    Function to train a model.

    Parameters:
    model (nn.Module): The model to train.
    epochs (int): The number of epochs to train for.
    validate_every (int): The number of iterations after which to validate the model.
    train_loader (DataLoader): The dataloader for the training data.
    val_loader (DataLoader): The dataloader for the validation data.
    lr (float): The learning rate.
    lr_decay (float): The learning rate decay.
    device (str): The device to use for computations.
    outp (str): The output directory to save the model.
    silent (bool): Whether to print progress or not.
    usefirstorder (bool): Whether to use first order optimization or not.
    kfacargs (KFACargs): The arguments for K-FAC.
    extrareg (function): The extra regularization function.
    wdecay (float): The weight decay.

    Returns:
    nn.Module, list, list, list: The trained model, the validation losses, the training losses, and the accuracies.
    """
    # Custom loss function to allow us to add direct l2 regularization to the second order model.
    def loss(x, y, inp):
        if wdecay > 0:
            return torch.sum(torch.square(x-y))/x.shape[0] + (
                                   wdecay*sum([torch.norm(x) for x in model.parameters()]) +
                                   (0 if extrareg is None else
                                   wdecay*0.5*torch.sum(torch.square(x-extrareg(inp)))/x.shape[0]))
        else:
            return torch.nn.MSELoss()(x,y)
    model.to(device)
    currentlr = lr
    kfacargs = kfacargs if kfacargs is not None else defaultKFACargs
    def optimf(lr):
        return optims.KFAC(lr, kfacargs.momentum, kfacargs.cov_ema_decay,
                           kfacargs.damping, kfacargs.stab_coeff,
                           use_cholesky=kfacargs.use_cholesky,
                           adjust_momentum=kfacargs.adjust_momentum)

    if not usefirstorder:
        try:
            import chainerkfac.optimizers as optims
            optim = optimf(lr)
        except:
            usefirstorder = True
    if usefirstorder:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
    train_losses = []
    val_losses = []
    val_its = []
    tqdmf = create_tqdm_bar
    if silent:
        tqdmf = lambda x, y: x
    for e in range(epochs):
        for i, x in enumerate(tqdmf(train_loader, "training")):
            data, target = x[0].to(device), x[1].squeeze().to(device)
            model.train()
            optim.zero_grad()
            output = model(data).squeeze()
            l = loss(output, target, data)
            train_losses.append(l.item())
            l.backward()
            optim.step()
            if usefirstorder:
                lr_scheduler.step()
            else:
                currentlr = currentlr * lr_decay
                optim.lr = currentlr
            it = i + e * len(train_loader)
            if validate_every > 0 and it % validate_every == 0:
                ls = validate_model(model, val_loader, nn.MSELoss(), device)
                vl = np.mean(ls)
                val_losses.append(vl)
                val_its.append(it)
                if not silent and it > 0:
                    plt.plot(train_losses)
                    plt.plot(val_its, val_losses)
                    plt.show()
                    print("\n", e, i, vl, np.std(ls), l.item())
                    if False:
                        save_path = time.strftime("%Y%m%d-%H%M%S")
                        path = f'{outp}/{save_path}_{e}/network.pth'
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        model.save(path)
    return val_losses, train_losses
