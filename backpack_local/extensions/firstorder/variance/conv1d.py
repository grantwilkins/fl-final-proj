from backpack_local.extensions.firstorder.gradient.conv1d import GradConv1d
from backpack_local.extensions.firstorder.sum_grad_squared.conv1d import SGSConv1d

from .variance_base import VarianceBaseModule


class VarianceConv1d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConv1d(),
            sgs_extension=SGSConv1d(),
        )
