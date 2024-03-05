from backpack_local.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack_local.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack_local.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPAvgPool2d(HMPBase):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class HMPMaxpool2d(HMPBase):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
