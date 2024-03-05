from backpack_local.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPZeroPad2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
