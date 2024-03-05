from backpack_local.core.derivatives.dropout import DropoutDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPDropout(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
