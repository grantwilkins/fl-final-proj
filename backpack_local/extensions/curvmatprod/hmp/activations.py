from backpack_local.core.derivatives.relu import ReLUDerivatives
from backpack_local.core.derivatives.sigmoid import SigmoidDerivatives
from backpack_local.core.derivatives.tanh import TanhDerivatives
from backpack_local.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPReLU(HMPBase):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class HMPSigmoid(HMPBase):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class HMPTanh(HMPBase):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
