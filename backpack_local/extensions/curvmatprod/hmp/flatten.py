from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPFlatten(HMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
