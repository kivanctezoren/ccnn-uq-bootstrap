import _init_paths

import logging as lg
import torch
from torchvision import datasets, transforms

from bootstrap import bootstrap
from ccnn import ccnn

# Globals for configuration:
BATCH_SIZE = 256
NUM_WORKERS = 1
DATASET_PATH = "datasets/"
# List of datasets that can be used
DATASETS_LIST = ["MNIST", "FashionMNIST", "CIFAR10"]
DATASET = "MNIST"
DATASET_TRAIN_CNT = -1
DATASET_TEST_CNT = 1000

lg.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)

if __name__ == "__main__":
    if DATASET not in DATASETS_LIST:
        raise ValueError("Given dataset is not supported: " + DATASET)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    lg.info("Using device:" + str(device))
    
    train_transform_list = [
        # transforms.RandomCrop(...)
    ]
    test_transform_list = []
    
    # Add tensor conversion
    train_transform_list.append(transforms.ToTensor())
    test_transform_list.append(transforms.ToTensor())
    
    # Compose transformations
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
    
    if DATASET == "MNIST":
        train_dset = datasets.MNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.MNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
    elif DATASET == "FashionMNIST":
        train_dset = datasets.FashionMNIST(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.FashionMNIST(DATASET_PATH, train=False, transform=test_transform, download=True)
    elif DATASET == "CIFAR10":
        train_dset = datasets.CIFAR10(DATASET_PATH, train=True, transform=train_transform, download=True)
        test_dset = datasets.CIFAR10(DATASET_PATH, train=False, transform=test_transform, download=True)
    else:
        raise ValueError("Unrecognized dataset name: " + DATASET)
    
    if DATASET_TRAIN_CNT > 0:
        train_dset = torch.utils.data.Subset(train_dset, range(0, DATASET_TRAIN_CNT))
    
    if DATASET_TEST_CNT > 0:
        test_dset = torch.utils.data.Subset(test_dset, range(0, DATASET_TEST_CNT))
    
    train_dl = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    ccnn_model = ccnn.CCNN(
        train_dl=train_dl,
        test_dl=test_dl,
        train_img_cnt=len(train_dset),
        test_img_cnt=len(test_dset),
        device=device
    )
    
    # TODO: Train CCNN
    ...
    
    # TODO: Apply bootstrap
    ...
