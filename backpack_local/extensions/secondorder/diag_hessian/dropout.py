from backpack_local.core.derivatives.dropout import DropoutDerivatives
from backpack_local.extensions.secondorder.diag_hessian.diag_h_base import (
    DiagHBaseModule,
)


class DiagHDropout(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
