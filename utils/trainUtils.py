import logging
import sys, os
import torch
import pickle
from dataloader import Dtfloader

def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr= lr, weight_decay = l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay = l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))

def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def getDataLoader(dataset:str, path):
    dataset = dataset.lower()
    if dataset == 'ali-ccp':
        return Dtfloader.Ali_CCPLoader(path)
    elif dataset == 'ali-mama':
        return Dtfloader.Ali_MamaLoader(path)
    else:
        raise ValueError("dataset {} not supported".format(dataset))


def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    return [i+1 for i in list(defaults.values())]

def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style= '{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger