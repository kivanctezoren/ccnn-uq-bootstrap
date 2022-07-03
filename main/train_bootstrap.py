import _init_paths
from bootstrap import bootstrap
from ccnn import ccnn

import torch
from torchvision import datasets, transforms

# Globals for configuration:
BATCH_SIZE = 256
NUM_WORKERS = 1
DATASET_PATH = "datasets/"
# List of datasets that can be used
DATASETS_LIST = ["MNIST", "FashionMNIST", "CIFAR10"]
DATASET = "MNIST"
DATASET_TRAIN_CNT = -1
DATASET_TEST_CNT = 1000

if __name__ == "__main__":
    if DATASET not in DATASETS_LIST:
        raise ValueError("Given dataset is not supported: " + DATASET)
    
    # No transformations done by default
    # TODO: Transformations might be applied throughout the CCNN code, check after implementation
    train_transform_list = []
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
        train_dset = torch.utils.data.Subset(train_dset, (0, DATASET_TRAIN_CNT))
    
    if DATASET_TEST_CNT > 0:
        # FIXME: test_dl has only two images. Is there a problem here?
        test_dset = torch.utils.data.Subset(test_dset, (0, DATASET_TEST_CNT))
    
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
        num_train=len(train_dset),
        num_test=len(test_dset)
    )
    
    # TODO: Train CCNN
    ...
    
    # TODO: Apply bootstrap
    ...
