from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPFlatten(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
