from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.secondorder.diag_hessian.diag_h_base import (
    DiagHBaseModule,
)


class DiagHFlatten(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
