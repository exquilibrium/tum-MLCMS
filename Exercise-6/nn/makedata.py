import numpy as np
import torch.utils.data

from linear_nn import *
import sys
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)
from utils.data import *
from utils.dataFrames import *
from training import *
def getfns(k,fn,odn):
    """
    Function to get the data and target file paths.

    Parameters:
    k (int): The index of the data.
    fn (str): The base name of the file.
    odn (str): The output directory.

    Returns:
    str, str: The data file path and the target file path.
    """
    return f"{odn}/{fn}/data{k}", f"{odn}/{fn}/targets{k}"

def getfilename(name):
    """
    Function to get the base name of a file without the extension.

    Parameters:
    name (str): The file path.

    Returns:
    str: The base name of the file without the extension.
    """
    return os.path.splitext(os.path.basename(name))[0]


def random_split(dataset, fracs):
    """
    Function to randomly split a dataset.

    Parameters:
    dataset (TensorDataset): The dataset to split.
    fracs (list): The fractions to split the dataset.

    Returns:
    list: The list of split datasets.
    """
    l = dataset.data.shape[0]
    inds = np.random.choice(np.arange(l), l, replace=False)
    cs = [0] + [int(l * x) for x in np.cumsum(fracs)]
    return [TensorDataset(dataset.data[inds[cs[i]:cs[i + 1]]], dataset.targets[inds[cs[i]:cs[i + 1]]]) for i in
            range(len(cs) - 1)]

def makedatatargets(datapre,k, args, removefewneighbors, everyxth, usefuture = 1, usepast = False):
    """
    Function to make data and targets.

    Parameters:
    datapre (str): The preprocessed data.
    k (int): The index of the data.
    args (list): The arguments.
    removefewneighbors (bool): Whether to remove few neighbors or not.
    everyxth (int): The interval to use for the data.
    usefuture (int): The number of future data points to use.
    usepast (bool): Whether to use past data points or not.

    Returns:
    list: The data and targets.
    """
    pedpos = getpedpos(datapre, everyxth, usefuture, usepast)
    return makedatafrompedpos(pedpos, k, args, removefewneighbors = removefewneighbors)

def makedatasetsseperate(datafn, targetsfn, k, files, args, removefewneighbors = False, everyxth = 8, usefuture = 16, usepast = False):
    """
    Function to make datasets separately.

    Parameters:
    datafn (str): The data file path.
    targetsfn (str): The target file path.
    k (int): The index of the data.
    files (list): The list of files.
    args (list): The arguments.
    removefewneighbors (bool): Whether to remove few neighbors or not.
    everyxth (int): The interval to use for the data.
    usefuture (int): The number of future data points to use.
    usepast (bool): Whether to use past data points or not.
    """
    for x in files:
        d = makedatatargets(transform(makedata(x)),k, args, removefewneighbors, everyxth, usefuture, usepast)
        np.save(datafn+getfilename(x),d[0])
        np.save(targetsfn+getfilename(x),d[1])

def loaddatasetsseperate(datafn, targetsfn, files):
    """
    Function to load datasets separately.

    Parameters:
    datafn (str): The data file path.
    targetsfn (str): The target file path.
    files (list): The list of files.

    Returns:
    np.array, np.array: The data and targets.
    """
    return (np.concatenate([np.load(datafn+getfilename(x)+".npy") for x in files]),
            np.concatenate([np.load(targetsfn+getfilename(x)+".npy") for x in files]))



def concatdataset(datasets):
    """
    Function to concatenate datasets.

    Parameters:
    datasets (list): The list of datasets to concatenate.

    Returns:
    TensorDataset: The concatenated dataset.
    """
    return TensorDataset(np.concatenate([x.data for x in datasets], 0),
                         np.concatenate([x.targets for x in datasets], 0))

experimetts = ["B/B","C/C","B/C","C/B","C+B/B","C+B/C","C+B/C+B"]

def getexperiment(datasetb, datasetc, experiment_type):
    """
    Function to create the experiments based on the type of the experiment.

    Parameters:
    datasetb (TensorDataset): The dataset B.
    datasetc (TensorDataset): The dataset C.
    experiment_type (str): The type of the experiment.

    Returns:
    TensorDataset, TensorDataset: The training/validation dataset and the test dataset.
    """
    # Depending on the experiment type, different datasets are used for training/validation and testing
    if experiment_type == "C/B":
        trainval_dataset, test_dataset = datasetc, datasetb
    if experiment_type == "C+B/C+B":
        trainval_dataset, test_dataset = torch.utils.data.random_split(
            torch.utils.data.ConcatDataset([datasetb, datasetc])
            , [0.5, 0.5])
    if experiment_type == "B/C":
        trainval_dataset, test_dataset = datasetb, datasetc
    if experiment_type == "C+B/B":
        trainval_datasetb, test_dataset = torch.utils.data.random_split(datasetb, [0.5, 0.5])
        f = len(trainval_datasetb) / len(datasetc)
        trainval_dataset = torch.utils.data.ConcatDataset([trainval_datasetb, torch.utils.data.random_split(datasetc, [f, 1 - f])[0]])
    if experiment_type == "C+B/C":
        f = len(datasetb) / len(datasetc)
        if f < 0.5:
            trainval_dataset, test_datasetc = torch.utils.data.random_split(datasetc, [f, f, 1 - 2 * f])[:2]
        else:
            trainval_dataset, test_datasetc = torch.utils.data.random_split(datasetc, [f, 1 - f])
        test_dataset = torch.utils.data.ConcatDataset([test_datasetc, datasetb])

    if experiment_type == "C/C":
        trainval_dataset, test_dataset = torch.utils.data.random_split(datasetc, [0.5, 0.5])
    if experiment_type == "B/B":
        trainval_dataset, test_dataset = torch.utils.data.random_split(datasetb, [0.5, 0.5])
    return trainval_dataset, test_dataset

def getexperimentRAM(datasetb, datasetc, experiment_type):
    """
    Function to create the experiments based on the type of the experiment.

    Parameters:
    datasetb (TensorDataset): The dataset B.
    datasetc (TensorDataset): The dataset C.
    experiment_type (str): The type of the experiment.

    Returns:
    TensorDataset, TensorDataset: The training/validation dataset and the test dataset.
    """
    # Depending on the experiment type, different datasets are used for training/validation and testing
    if experiment_type == "C/B":
        trainval_dataset, test_dataset = datasetc, datasetb
    if experiment_type == "C+B/C+B":
        trainval_dataset, test_dataset = random_split(
            concatdataset([datasetb, datasetc])
            , [0.9, 0.1])
    if experiment_type == "B/C":
        trainval_dataset, test_dataset = datasetb, datasetc
    if experiment_type == "C+B/B":
        trainval_datasetb, test_dataset = random_split(datasetb, [0.9, 0.1])
        f = len(trainval_datasetb) / len(datasetc)
        trainval_dataset = concatdataset([trainval_datasetb, random_split(datasetc, [f, 1 - f])[0]])
    if experiment_type == "C+B/C":
        f = len(datasetb) / len(datasetc)
        if f < 0.5:
            trainval_dataset, test_datasetc = random_split(datasetc, [f, f, 1 - 2 * f])[:2]
        else:
            trainval_dataset, test_datasetc = random_split(datasetc, [f, 1 - f])
        test_dataset = concatdataset([test_datasetc, datasetb])

    if experiment_type == "C/C":
        trainval_dataset, test_dataset = random_split(datasetc, [0.9, 0.1])
    if experiment_type == "B/B":
        trainval_dataset, test_dataset = random_split(datasetb, [0.9, 0.1])
    return trainval_dataset, test_dataset


def weidmann(sk, targ):
    return targ["v"] * (1 - np.exp((targ["l"] - sk) / (targ["v"] * targ["t"])))


def getmse(data, labels, targ):
    return np.average(np.square(weidmann(data, targ) - labels))

def makegetdata(bottleneck_files, corridor_files, odn, arg, k, experiment_type, targ):
    """
    Make data using the original method and save for future use.
    Parameters:
    bottleneck_files (list<str>): Files for dataset B.
    corridor_files (list<str>): Files for dataset C.
    experiment_type (str): The type of the experiment.
    arg (ArgsClass): Object describing what features to use. (NN1 NN2 ...)
    k (int): Number of closest neighbors to use.
    experiment_type (str): What experiments to use for which splits.
    targ (TargClass): Which

    Returns:
    TensorDataset, TensorDataset, int: The training/validation dataset and the test dataset and the target size.
    """
    argstring = f"{'1' if arg.use_rel_vel else '0'}{'1' if arg.use_mean_dist_spacing else '0'}"
    directory = f"{odn}{argstring}/{k}"
    datafn, targetsfn = f"{directory}/data", f"{directory}/targets"
    os.makedirs(f"{directory}", exist_ok=True)
    if not os.path.isfile(datafn + getfilename(bottleneck_files[0]) + ".npy"):
        makedatasetsseperate(datafn, targetsfn, k,
                             bottleneck_files + corridor_files, arg, everyxth=targ.everyxth, usefuture=targ.usefuture,
                             usepast=targ.usepast)
    datac, labelsc = loaddatasetsseperate(datafn, targetsfn, corridor_files)
    datab, labelsb = loaddatasetsseperate(datafn, targetsfn, bottleneck_files)
    datasetc = TensorDataset(datac, labelsc)
    datasetb = TensorDataset(datab, labelsb)
    return getexperimentRAM(datasetb, datasetc, experiment_type) + (1,)

def defaultdataf(bottleneck_files, corridor_files, odn, arg, k, experiment_type, targ):
    """
    Make velocity data using the method described in task2 part1.
    Parameters:
    bottleneck_files (list<str>): Files for dataset B.
    corridor_files (list<str>): Files for dataset C.
    experiment_type (str): The type of the experiment.
    arg (ArgsClass): Object describing what features to use. (NN1 NN2 ...)
    k (int): Number of closest neighbors to use.
    experiment_type (str): What experiments to use for which splits.
    targ (TargClass): Which

    Returns:
    TensorDataset, TensorDataset, int: The training/validation dataset and the test dataset and the target size.
    """
    os.makedirs(odn, exist_ok=True)
    fnb = f"{odn}/datab"
    fnc = f"{odn}/datac"
    if os.path.isfile(fnb+".npy"):
        datab = np.load(fnb+".npy")
    else:
        datab = getdatapoints(bottleneck_files,arg,k)
    if os.path.isfile(fnc+".npy"):
        datac = np.load(fnc+".npy")
    else:
        datac = getdatapoints(corridor_files,arg,k)
    np.save(fnb,datab)
    np.save(fnc,datac)
    datab = datab[np.arange(0,datab.shape[0],targ.everyxth)] /100
    datac = datac[np.arange(0,datac.shape[0],targ.everyxth)] /100
    datasetb = TensorDataset(datab[:,:-2],datab[:,-2:])
    datasetc = TensorDataset(datac[:,:-2],datac[:,-2:])
    return getexperimentRAM(datasetb,datasetc,experiment_type) + (2,)

def defaultdatafoneoutput(bottleneck_files, corridor_files, odn, arg, k, experiment_type, targ):
    """
    Make velocity data using the method described in task2 part1 and then calculate speed from mean velocity.
    Parameters:
    bottleneck_files (list<str>): Files for dataset B.
    corridor_files (list<str>): Files for dataset C.
    experiment_type (str): The type of the experiment.
    arg (ArgsClass): Object describing what features to use. (NN1 NN2 ...)
    k (int): Number of closest neighbors to use.
    experiment_type (str): What experiments to use for which splits.
    targ (TargClass): Which

    Returns:
    TensorDataset, TensorDataset, int: The training/validation dataset and the test dataset and the target size.
    """
    os.makedirs(odn, exist_ok=True)
    fnb = f"{odn}/datab"
    fnc = f"{odn}/datac"
    if os.path.isfile(fnb+".npy"):
        datab = np.load(fnb+".npy")
    else:
        datab = getdatapoints(bottleneck_files,arg,k)
    if os.path.isfile(fnc+".npy"):
        datac = np.load(fnc+".npy")
    else:
        datac = getdatapoints(corridor_files,arg,k)
    np.save(fnb,datab)
    np.save(fnc,datac)
    datab = datab[np.arange(0,datab.shape[0],targ.everyxth)] /100
    datac = datac[np.arange(0,datac.shape[0],targ.everyxth)] /100
    datasetb = TensorDataset(datab[:,:-2], np.sqrt(np.sum(np.square(datab[:,-2:]),axis=1)))
    datasetc = TensorDataset(datac[:,:-2], np.sqrt(np.sum(np.square(datac[:,-2:]),axis=1)))
    return getexperimentRAM(datasetb,datasetc,experiment_type) + (1,)