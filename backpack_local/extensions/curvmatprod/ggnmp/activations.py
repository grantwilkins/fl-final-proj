from backpack_local.core.derivatives.relu import ReLUDerivatives
from backpack_local.core.derivatives.sigmoid import SigmoidDerivatives
from backpack_local.core.derivatives.tanh import TanhDerivatives
from backpack_local.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPReLU(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class GGNMPSigmoid(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class GGNMPTanh(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
