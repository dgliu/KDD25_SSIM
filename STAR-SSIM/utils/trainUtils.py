import logging
import sys, os
import pickle
from dataloader import Dtfloader

def getDataLoader(dataset:str, path):
    dataset = dataset.lower()
    if dataset == 'ali-ccp':
        return Dtfloader.Ali_CCPLoader(path)
    elif dataset == 'ali-mama':
        return Dtfloader.Ali_MamaLoader(path)

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