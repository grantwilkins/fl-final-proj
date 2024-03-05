from backpack_local.core.derivatives.conv3d import Conv3DDerivatives
from backpack_local.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConv3d(SGSBase):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
