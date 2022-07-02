import torchvision

from lib.bootstrap import bootstrap
from lib.ccnn import CCNN


DATASET_PATH = "datasets/"
# Datasets that can be used
DATASETS_LIST = ["MNIST", "FashionMNIST", "CIFAR10"]
DATASET = "MNIST"

if __name__ == "__main__":
    if DATASET not in DATASETS_LIST:
        raise ValueError("Given dataset is not supported: " + DATASET)
    
    # No transformations done by default
    train_transform = None
    test_transform = None
    
    if DATASET == "MNIST":
        train_dset = torchvision.datasets.MNIST(DATASET_PATH, train=True, transform=train_transform)
        test_dset = torchvision.datasets.MNIST(DATASET_PATH, train=False, transform=test_transform)
    elif DATASET == "FashionMNIST":
        train_dset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=True, transform=train_transform)
        test_dset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=False, transform=test_transform)
    elif DATASET == "CIFAR10":
        train_dset = torchvision.datasets.CIFAR10(DATASET_PATH, train=True, transform=train_transform)
        test_dset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, transform=test_transform)
    else:
        raise ValueError("Unrecognized dataset name: " + DATASET)
    
    ccnn = CCNN(...)
    
    # TODO: Train CCNN
    ...
    
    # TODO: Apply bootstrap
    ...
