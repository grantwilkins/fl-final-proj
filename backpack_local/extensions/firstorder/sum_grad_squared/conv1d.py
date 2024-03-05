from backpack_local.core.derivatives.conv1d import Conv1DDerivatives
from backpack_local.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConv1d(SGSBase):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
