"""Define our models, and training and eval functions."""

from torch import nn
from project.types.common import NetGen

Net = nn.Sequential(
    nn.Conv2d(1, 32, 5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), padding=1),
    nn.Conv2d(32, 64, 5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), padding=1),
    nn.Flatten(1),
    nn.Linear(64 * 7 * 7, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

get_net: NetGen = lambda x, y: Net  # noqa: E731,ARG005
# ResNet18 = ResNet(img_channels=3, num_layers=18, num_classes=10)

# get_net: NetGen = lambda x, y: ResNet18
