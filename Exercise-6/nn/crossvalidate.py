import numpy as np
import torch
import os
import sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)
from utils.data import *
from makedata import *
from training import *


class Argsclass:
    """
    Class to hold arguments for the dataset.

    Attributes:
    use_rel_vel (bool): Whether to use relative velocity or not.
    use_mean_dist_spacing (bool): Whether to use mean distance spacing or not.
    sizef (function): Function to calculate the size of the data.
    """

    def __init__(self, use_rel_vel=False, use_mean_dist_spacing=False):
        self.use_rel_vel = use_rel_vel
        self.use_mean_dist_spacing = use_mean_dist_spacing
        # sizef function calculates the size based on whether relative velocity and mean distance spacing are used or not
        self.sizef = lambda x: x * (4 if self.use_rel_vel else 2) + (1 if self.use_mean_dist_spacing else 0)


args = [Argsclass(), Argsclass(True, False), Argsclass(False, True), Argsclass(True, True)]


def makekbtargs(ls, ts, vs):
    targs = []
    for x in ls:
        for y in ts:
            for z in vs:
                targs.append({"l": x, "t": y, "v": z})
    return targs


def getbestkb(datatr, labelstr, targs, silent):
    """
    Grid search Weidman model parameters.
    """
    scoredtargs = []
    for targ in targs:
        scoredtargs.append((targ, getmse(datatr, labelstr, targ)))
    if not silent:
        print(np.unique([x[1] for x in scoredtargs]))
        # print(np.array(scoredtargs)[np.argsort([x[1] for x in scoredtargs])[0:5]])
    besti = np.argmin([x[1] for x in scoredtargs])
    return scoredtargs[besti][0]


def showlatent(model, device, dataf):
    """
    Function to visualize the latent space of a model for given data.
    If the latent space is 2D or 3D, it will be plotted. Otherwise, a message will be printed.

    Parameters:
    model (nn.Module): The model to visualize the latent space of.
    device (torch.device): The device where the model and data are.
    dataf (function): The function to get the data.

    """
    # Get the data for "bottleneck" and "corridor"
    datab, targetsb = dataf("bottleneck")
    datac, targetsc = dataf("corridor")

    # Convert the data to PyTorch tensors and move them to the device
    datab = torch.from_numpy(datab).float().to(device)
    datac = torch.from_numpy(datac).float().to(device)

    # Set the model to evaluation mode
    model.eval()

    # Get the latent representations of the data without tracking gradients
    with torch.no_grad():
        latentb = model.getlatent(datab).cpu().numpy()
        latentc = model.getlatent(datac).cpu().numpy()

    # If the latent space is 2D, plot it
    if latentb.shape[1] == 2:
        plt.scatter(latentb[:, 0], latentb[:, 1], c=targetsb)
        plt.show()
        plt.scatter(latentc[:, 0], latentc[:, 1], c=targetsc)
        plt.show()
    # If the latent space is 3D, plot it
    elif latentb.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(latentb[:, 0], latentb[:, 1], latentb[:, 2], c=targetsb)
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(latentc[:, 0], latentc[:, 1], latentb[:, 2], c=targetsc)
        plt.show()
    # If the latent space has more than 3 dimensions, print a message
    else:
        print(f"{latentb.shape} too many dims")


def testloss(outsize):
    if outsize == 1:
        return nn.MSELoss()
    else:
        return nn.MSELoss()
        def lengthdist(x, y, z):
            print(x.shape)
            xlen = np.sqrt(np.sum(np.square(x.numpy()), 1))
            ylen = np.sqrt(np.sum(np.square(y.numpy()),1))
            np.average(np.square(xlen - ylen))
        return lengthdist


def crossvalidate(odn, arg, hs, targs, lr, epochs=2, experiment_type="C+B/B",
                  usekbreg=False, wdecay=0.2, dataf=None):
    """
    Function to perform cross-validation on a model with given parameters.

    Parameters:
    odn (str): The directory where data will be loaded from using dataf.
    arg (argsClass): The architecture arguments.
    hs (list): The hidden sizes.
    targs (list): The arguments to try.
    lr (float): The learning rate.
    epochs (int): The epoch count.
    experiment_type (str): The type of experiment one of ["B/B","C/C","B/C","C/B","C+B/B","C+B/C","C+B/C+B"].
    usekbreg (bool): Use knowledge based regularizer with strength of half of weight decay.
    wdecay (float): The strength of weight decay.
    dataf (function): The function to get data.

    Returns:
    tuple: The best targ descriptor, the best model.
    """
    # Initialize the best loss, best model and best descriptor
    bestlossoverall = np.infty
    bestmodeloverall = None
    bestdesc = None

    # Iterate over the arguments to try
    for targ in targs:
        # Get the training/validation and test datasets
        if dataf is None:
            trainval_dataset, test_dataset, outsize = makegetdata(bottleneck_files, corridor_files, odn, arg, 10,
                                                                  experiment_type, targ)
        else:
            trainval_dataset, test_dataset, outsize = dataf(bottleneck_files, corridor_files, odn,
                                                            arg, 10, experiment_type, targ)
        print(len(trainval_dataset), print(trainval_dataset.data.shape))

        # Split the training/validation dataset into folds for cross-validation
        foldsn = 5
        folds = random_split(trainval_dataset, [1 / foldsn for x in range(foldsn)])

        # Initialize the best loss and best model for this set of arguments
        bestloss = np.infty
        bestmodel = None

        # Determine the device to use for training
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Iterate over the folds for cross-validation
        for vali in range(len(folds)):
            # Create the training and validation datasets for this fold
            train_dataset = concatdataset(
                [folds[i] for i in range(len(folds)) if vali != i])
            val_dataset = folds[vali]

            # If using knowledge based regularizer, get the best knowledge based regularizer
            if usekbreg:
                targskb = makekbtargs(np.linspace(0.3, 1, 30),
                                      np.linspace(0.3, 1, 30),
                                      np.linspace(0.3, 2, 30))
                kbreg = getbestkb(train_dataset.data[:, -1], train_dataset.targets, targskb, True)

            # Create the data loaders for training and validation
            ltd = len(train_dataset)
            train_loader = train_dataset.makedataloader(targ.bsd)
            val_loader = test_dataset.makedataloader(targ.bsd)

            # Create the model
            if targ.usevea:
                model = LinearAE(hs[0], arg.sizef(10), outsize, lambda: nn.Sigmoid(), device)
            else:
                model = LinearNN(hs, arg.sizef(10), outsize, lambda: nn.Sigmoid())
            f = (lambda x: kbreg["v"] * (1 - torch.exp(
                (kbreg["l"] - x[:, -1]) / (kbreg["v"] * kbreg["t"])))) if usekbreg else None

            # Train the model
            losses, train_losses = train(model, epochs, max(1, len(train_loader) // 4),
                                         train_dataset.makedataloader() if targ.bsd == 1 else train_loader, val_loader,
                                         lr, targ.lrd, device,
                                         silent=targ.silent, usefirstorder=targ.usefirstorder,
                                         kfacargs=targ.optimarg,
                                         extrareg=f, wdecay=wdecay)
            lastloss = losses[-1]
            train_loss = train_losses[-1]
            if lastloss < bestloss:
                bestloss = lastloss
                bestmodel = model
            if True:
                test_loader = test_dataset.makedataloader(1000)
                print(len(test_loader))
                losses = validate_model(model, test_loader, testloss(outsize), device)
                print((hs, np.average(losses), np.std(losses),
                       lastloss, train_loss, str(targ)))

        # Validate the best model on the test set
        test_loader = test_dataset.makedataloader(1000)
        losses = validate_model(bestmodel, test_loader, testloss(outsize), device)
        if np.mean(losses) < bestlossoverall:
            bestlossoverall = np.mean(losses)
            bestdesc = (hs, np.average(losses), np.std(losses), str(targ))
            bestmodeloverall = bestmodel

    # Return the best descriptor and best model
    return bestdesc, bestmodeloverall


class TargClass:
    """
    Class describing the training and data processing procedure.
    """

    def __init__(self, bsd, lrd, silent, usefirstorder, optimarg, usevea, everyxth=5, usefuture=5, usepast=False):
        (self.bsd, self.lrd, self.silent,
         self.usefirstorder, self.optimarg, self.usevea) = (bsd, lrd, silent,
                                                            usefirstorder, optimarg, usevea)
        """
            Function to perform cross-validation on a model with given parameters.

            Parameters:
            bsd (int): The numer of batched to split the dataset into.
            lrd (float): The learning rate decay.
            usefirstorder (bool): Whether to use a first order optimizer.
            optimarg (KFACargs): The arguments of the second order optimizer.
            usevea (bool): Use variational encoder with its hidden size equal to the last hidden layer size.
            everyxth (int): Use every xth datapoint.
            usefuture (int): How far to the future to look for the second position for speed and velocity.
            usepast (bool): If set to true we look usefuture into the past and the future and use the average interframe speed in this window.
            """

        self.everyxth = everyxth
        self.usefuture = usefuture
        self.usepast = usepast

    def __str__(self):
        return str((self.bsd, self.lrd, self.silent, self.usefirstorder, self.usevea, str(self.optimarg)))


def maketargssimple(bsds, lrds, usefirstorder):
    targs = []
    for x in bsds:
        for y in lrds:
            targs.append(TargClass(x, y, True, usefirstorder, None, False))
    return targs


def maketargs(bsds, lrds, silent, usefirstorder, optimargs, usevae, everyxth=80, usefuture=8, usepast=True):
    targs = []
    for x in bsds:
        for y in lrds:
            for z in optimargs:
                targs.append(TargClass(x, y, silent, usefirstorder, z, usevae, everyxth, usefuture, usepast))
    return targs
