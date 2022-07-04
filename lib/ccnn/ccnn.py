import logging as lg
import math
import torch.utils.data

from numpy import linalg as la
from sklearn.preprocessing import label_binarize
from .utils import get_pixel_vector, zca_whitening, transform_and_pooling, low_rank_matrix_regression

# CCNN is defined for two layers, additional methods are required for adding more.
# Strings indicating the methods:
MULTILAYER_METHODS = ["ZHANG", "TRANSFER_LRN"]
# "ZHANG": Layer generation method proposed in the original CCNN paper by Zhang et al.
# "TRANSF_LRN": The transfer learning method proposed by Du et al.

lg.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)


class CCNN:
    def __init__(self,
                 train_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 train_img_cnt: int,
                 test_img_cnt: int,
                 multilayer_method: str = "ZHANG",
                 device: torch.device = torch.device("cpu")
                 ):
        if multilayer_method not in MULTILAYER_METHODS:
            raise ValueError("Unrecognized CCNN layer addition method: " + multilayer_method)
        
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.train_img_cnt = train_img_cnt
        self.test_img_cnt = test_img_cnt
        
        self.device = device
        
        self.layer_count = 0
        self.img_cnt = self.train_img_cnt + self.test_img_cnt

        self.filter_weight = None
        self.last_layer_output = None
        
        if multilayer_method == "ZHANG":
            # Generate first layer
            self.generate_layer()  # Increments layer_count too
        elif multilayer_method == "TRANSFER_LRN":
            # TODO: Take a pretrained model and use its state
            
            self.layer_count += 1
        else:
            raise ValueError("Unrecognized CCNN layer addition method in first layer generation: " + multilayer_method)

    # TODO: Save reduced input & learned features
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
        Train and add a layer to the CCNN with the method proposed by Zhang et al. in paper ...
        
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
        x_train = []
        x_test = []
        labels = []
        
        for inp, lbl in self.train_dl:
            x_train.append(inp)
            labels.append(lbl)
        
        for inp, lbl in self.test_dl:
            x_test.append(inp)
            labels.append(lbl)
        
        # Process train & test toghether
        x_train = torch.vstack(x_train).to(self.device)
        x_test = torch.vstack(x_test).to(self.device)
        
        x_raw = torch.cat((x_train, x_test)).to(self.device)
        labels = torch.hstack(labels).to(self.device)

        lg.debug("x_train shape: " + str(x_train.shape))
        lg.debug("x_test shape: " + str(x_test.shape))
        lg.debug("x_raw shape (2D): " + str(x_raw.shape))
        lg.debug("labels shape: " + str(labels.shape))
        
        lg.info("Detecting image parameters...")
        if x_raw.shape[2] != x_raw.shape[3]:
            raise ValueError(f"Expected square images, instead got width {x_raw.shape[2]}, height {x_raw.shape[3]}.")
        
        img_size = x_raw.shape[2]  # == x_raw.shape[3] == L (one side of img.)
        patch_size = 1 + (patch_radius * 2)  # F = 2*radius + 1
        patch_pixel_cnt = patch_size ** 2
        
        patch_cnt_one_side = img_size - (patch_radius * 2)  # == (L - F + 1)
        patch_cnt = patch_cnt_one_side ** 2
        
        pool_cnt = (patch_cnt_one_side // pooling_stride) ** 2
        
        channel_cnt = x_raw.shape[1]
        # feature_dim = nystrom_dim  # Since there is a single channel ...?

        # Vectorize the inputs
        # TODO: Check whether this should be done
        x_raw = x_raw.reshape((x_raw.shape[0], x_raw.shape[1], 1, -1)).squeeze(2)

        lg.debug("x_raw shape (vectorized 1D): " + str(x_raw.shape))

        lg.info("Constructing the patches...")
        
        patch = torch.zeros((self.img_cnt, patch_cnt, channel_cnt, patch_pixel_cnt),
                            dtype=torch.float32,
                            device=self.device)
        
        for y in range(0, patch_cnt_one_side):
            for x in range(0, patch_cnt_one_side):
                for i in range(0, channel_cnt):
                    indices = get_pixel_vector(x + patch_radius, y + patch_radius, patch_radius, img_size)
                    patch[:, x + y * patch_cnt_one_side, i] = x_raw[:, i, indices]
        
        lg.debug("patch shape: " + str(patch.shape))
        
        lg.info("Applying local contrast normalization and ZCA whitening...")
        patch = patch.reshape((self.img_cnt * patch_cnt, channel_cnt * patch_pixel_cnt))
        patch -= torch.mean(patch, axis=1).reshape((patch.shape[0], 1))
        patch /= la.norm(patch, axis=1).reshape(patch.shape[0], 1) + 0.1
        patch = zca_whitening(patch)
        patch = patch.reshape((self.img_cnt, patch_cnt, channel_cnt, patch_pixel_cnt))

        lg.debug("patch shape after normalization & whitening: " + str(patch.shape))
        
        lg.info("Creating features...")
        transformer = [0]
        base = 0
        feature_dim = nystrom_dim
        x_reduced = torch.zeros((self.img_cnt, pool_cnt * feature_dim), dtype=torch.float16, device=self.device)
        
        while base < self.img_cnt:
            lg.debug("Sample ID: " + str(base) + "-" + str(min(self.img_cnt, base + chunk_size)))
            x_reduced[base:min(self.img_cnt, base + chunk_size)], transformer = transform_and_pooling(
                patch=patch[base:min(self.img_cnt, base + chunk_size)],
                transformer=transformer,
                selected_group_size=[channel_cnt],  # Always 1?
                gamma=gamma,
                nystrom_dim=nystrom_dim,
                patch_per_side=patch_cnt_one_side,
                pooling_size=pooling_size,
                pooling_stride=pooling_stride
            )
        
        lg.info("Applying normalization...")
        x_reduced = x_reduced.reshape((self.img_cnt * pool_cnt, feature_dim))
        x_reduced -= torch.mean(x_reduced, axis=0)
        x_reduced /= la.norm(x_reduced) / math.sqrt(self.img_cnt * pool_cnt)
        x_reduced = x_reduced.reshape((self.img_cnt, pool_cnt * feature_dim))
        
        lg.info("Learning filters...")
        labels_binarized = label_binarize(labels, classes=range(0, 10))
        filter_weight = low_rank_matrix_regression(
            # Split and pass concatenated sets
            x_train=x_reduced[0:self.train_img_cnt],
            y_train=labels_binarized[0:self.train_img_cnt],
            x_test=x_reduced[self.train_img_cnt:],
            y_test=labels_binarized[self.train_img_cnt:],
            
            d1=pool_cnt,
            d2=feature_dim,
            n_iter=n_iter,
            reg=regularization_param,
            learning_rate=learning_rate,
            ratio=crop_ratio
        )
    
        filter_dim = filter_weight.shape[0]
        
        lg.info("Applying filters...")
        output = torch.dot(x_reduced.reshape((self.img_cnt * pool_cnt, feature_dim)), filter_weight.T)
        output = torch.reshape(output, (self.img_cnt, filter_dim))
        output = torch.transpose(output, 1, 2)  # Transpose last 2 dimensions
        
        lg.info("Feature dimension: " + str(output[0].size))
        lg.debug("output shape:", output.shape)
        
        self.layer_count += 1
        self.filter_weight = filter_weight
        self.last_layer_output = output
        
        lg.info("Done layer generation #" + str(self.layer_count) + ".")
        
    def forward(self, inp):
        if self.filter_weight is None:
            raise Exception("The CCNN has got no layers.")
        
        return torch.dot(inp, self.filter_weight.T)
