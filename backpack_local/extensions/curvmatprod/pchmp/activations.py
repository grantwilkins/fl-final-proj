from backpack_local.core.derivatives.relu import ReLUDerivatives
from backpack_local.core.derivatives.sigmoid import SigmoidDerivatives
from backpack_local.core.derivatives.tanh import TanhDerivatives
from backpack_local.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPReLU(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class PCHMPSigmoid(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class PCHMPTanh(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
