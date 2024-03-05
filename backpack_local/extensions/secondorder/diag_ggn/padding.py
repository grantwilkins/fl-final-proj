from backpack_local.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack_local.extensions.secondorder.diag_ggn.diag_ggn_base import (
    DiagGGNBaseModule,
)


class DiagGGNZeroPad2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
