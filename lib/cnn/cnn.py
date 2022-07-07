import torch.nn.functional as F
from torch import nn


class LeNet(nn.Module):
    """ A LeNet model, supporting parameters described in the paper by Du et al."""
    
    def __init__(self, n_classes, paper_params=False):
        super(LeNet, self).__init__()
        
        if paper_params:  # Use parameters described in the paper
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
                nn.Tanh()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(in_features=128, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )
        else:  # Use parameters giving better results on MNIST dataset
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1),
                nn.Tanh()
            )
    
            self.classifier = nn.Sequential(
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=n_classes),
            )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
