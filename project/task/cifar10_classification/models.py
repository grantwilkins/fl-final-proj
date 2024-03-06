"""Define our models, and training and eval functions."""

from project.task.cifar10_classification.resnet18 import ResNet
from project.types.common import NetGen

Net = ResNet(img_channels=3, num_layers=18, num_classes=10)

get_net: NetGen = lambda x, y: Net  # noqa: E731,ARG005
