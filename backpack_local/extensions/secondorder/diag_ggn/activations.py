from backpack_local.core.derivatives.elu import ELUDerivatives
from backpack_local.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack_local.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack_local.core.derivatives.relu import ReLUDerivatives
from backpack_local.core.derivatives.selu import SELUDerivatives
from backpack_local.core.derivatives.sigmoid import SigmoidDerivatives
from backpack_local.core.derivatives.tanh import TanhDerivatives
from backpack_local.extensions.secondorder.diag_ggn.diag_ggn_base import (
    DiagGGNBaseModule,
)


class DiagGGNReLU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class DiagGGNSigmoid(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class DiagGGNTanh(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())


class DiagGGNELU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class DiagGGNSELU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())


class DiagGGNLeakyReLU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class DiagGGNLogSigmoid(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
