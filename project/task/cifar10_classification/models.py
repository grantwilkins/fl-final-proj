"""Define our models, and training and eval functions."""

# from torchvision.models import resnet18
from project.types.common import NetGen
from project.task.cifar10_classification.resnet18 import ResNet

# Net = resnet18(num_classes=10)

Net = ResNet(img_channels=3, num_layers=18, num_classes=10)

get_net: NetGen = lambda x, y: Net
# get_net: NetGen = lazy_config_wrapper(Net)
