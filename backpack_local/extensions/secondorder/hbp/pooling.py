from backpack_local.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack_local.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack_local.extensions.secondorder.hbp.hbpbase import HBPBaseModule
from backpack_local.core.derivatives.adaptive_avg_pool_nd import (
    AdaptiveAvgPoolNDDerivatives,
)


class HBPAvgPool2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class HBPMaxpool2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class HBPAdaptiveAvgPoolNd(HBPBaseModule):
    def __init__(self, N: int):
        super().__init__(derivatives=AdaptiveAvgPoolNDDerivatives(N=N))
