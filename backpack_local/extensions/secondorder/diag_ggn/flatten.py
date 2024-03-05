from backpack_local.core.derivatives.flatten import FlattenDerivatives
from backpack_local.extensions.secondorder.diag_ggn.diag_ggn_base import (
    DiagGGNBaseModule,
)


class DiagGGNFlatten(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
