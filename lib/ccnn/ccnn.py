#from math_utils import ...

import logging as lg
import torch.utils.data

# CCNN is defined for two layers, additional methods are required for adding more.
# Strings indicating the methods:
MULTILAYER_METHODS = ["ZHANG", "TRANSFER_LRN"]
# "ZHANG": Layer generation method proposed in the original CCNN paper by Zhang et al.
# "TRANSF_LRN": The transfer learning method proposed by Du et al.

lg.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.INFO)


class CCNN:
    def __init__(self,
                 train_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 num_train: int,
                 num_test: int,
                 multilayer_method: str = "ZHANG"
                 ):
        if multilayer_method not in MULTILAYER_METHODS:
            raise ValueError("Unrecognized CCNN layer addition method: " + multilayer_method)
        
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.num_train = num_train
        self.num_test = num_test

        self.layer_count = 0
        self.n = self.num_train + self.num_test
        
        if multilayer_method == "ZHANG":
            # Generate first layer
            self.generate_layer()  # Increments layer_count too
        elif multilayer_method == "TRANSFER_LRN":
            # TODO: Take a pretrained model and use its state
            
            self.layer_count += 1
        else:
            raise ValueError("Unrecognized CCNN layer addition method in first layer generation: " + multilayer_method)
        
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
        """
        Train and add a layer to the CCNN.
        
        :param patch_radius: ...
        :param nystrom_dim: ...
        :param pooling_size: ...
        :param pooling_stride: ...
        :param gamma: ...
        :param regularization_param: ...
        :param learning_rate: ...
        :param crop_ratio: ...
        :param n_iter: ...
        :param chunk_size: ...
        :return: None.
        """
        
        lg.info("Begin generating layer #" + str(self.layer_count + 1) + ".")
        
        lg.info("Reading the dataset...")
        x_train = ...
        x_test = ...
        
        x_raw = ...
        label = ...
        
        lg.info("Detecting image parameters...")
        ...
        
        lg.info("Constructing the patches...")
        ...
        
        lg.info("Applying local contrast normalization and ZCA whitening...")
        ...
        
        lg.info("Creating features...")
        ...
        
        lg.info("Applying normalization...")
        ...
        
        lg.info("Learning filters...")
        ...
        
        lg.info("Applying filters...")
        ...
        
        self.layer_count += 1
        
        lg.info("Done layer generation #" + str(self.layer_count + 1) + ".")
