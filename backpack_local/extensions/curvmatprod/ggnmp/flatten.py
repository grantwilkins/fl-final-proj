from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPFlatten(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
