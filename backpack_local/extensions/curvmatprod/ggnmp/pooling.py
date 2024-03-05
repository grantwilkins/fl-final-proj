from backpack_local.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack_local.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPAvgPool2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class GGNMPMaxpool2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
