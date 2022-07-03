#from math_utils import ...
import torch.utils.data


class CCNN:
    def __init__(self,
                 train_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 num_train: int,
                 num_test: int,
                 ):
        self.layer_count = 0
        
        self.generate_layer()  # Increments layer_count too
        
    def generate_layer(self,
                       patch_radius: int = 2,
                       nystrom_dim: int = 200,
                       pooling_size: int = 2,
                       pooling_stride: int = 2,
                       gamma: float = 2.,
                       regularization_param: int = 100.,
                       learning_rate: float = 0.2,
                       crop_ratio: float = 1.,
                       n_iter: int = 5000,
                       chunk_size: int = 5000,
                       # max_channel: int = 16
                       ):
        ...
        
        self.layer_count += 1
