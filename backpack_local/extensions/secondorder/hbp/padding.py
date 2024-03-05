from backpack_local.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack_local.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPZeroPad2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
