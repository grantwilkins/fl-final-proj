from backpack_local.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack_local.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPZeroPad2d(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
