from backpack_local.core.derivatives.dropout import DropoutDerivatives
from backpack_local.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPDropout(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
