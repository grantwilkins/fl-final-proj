"""Define our models, and training and eval functions."""

from torchvision.models import resnet18
from project.utils.utils import lazy_config_wrapper
from project.types.common import NetGen

Net = resnet18

get_net: NetGen = lazy_config_wrapper(Net)
