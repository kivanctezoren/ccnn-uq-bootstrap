import _init_paths

import torch
import torchvision

import bootstrap
import ccnn


DATASET_PATH = "datasets/"
# Datasets that can be used
DATASETS_LIST = ["MNIST", "FashionMNIST", "CIFAR10"]
DATASET = "MNIST"
DATASET_TRAIN_CNT = -1
DATASET_TEST_CNT = 1000

if __name__ == "__main__":
    if DATASET not in DATASETS_LIST:
        raise ValueError("Given dataset is not supported: " + DATASET)
    
    # No transformations done by default
    train_transform = None
    test_transform = None
    
    if DATASET == "MNIST":
        train_dset = torchvision.datasets.MNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = torchvision.datasets.MNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
    elif DATASET == "FashionMNIST":
        train_dset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
    elif DATASET == "CIFAR10":
        train_dset = torchvision.datasets.CIFAR10(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, transform=test_transform, download=True)
    else:
        raise ValueError("Unrecognized dataset name: " + DATASET)
    
    if DATASET_TRAIN_CNT > 0:
        train_dset = torch.utils.data.Subset(train_dset, (0, DATASET_TRAIN_CNT))
    
    if DATASET_TEST_CNT > 0:
        test_dset = torch.utils.data.Subset(test_dset, (0, DATASET_TEST_CNT))
    
    train_dl = torch.utils.data.DataLoader(...)
    test_dl = torch.utils.data.DataLoader(...)
    
    ccnn_model = ccnn.CCNN(...)
    
    # TODO: Train CCNN
    ...
    
    # TODO: Apply bootstrap
    ...
