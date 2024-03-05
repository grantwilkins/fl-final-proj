from backpack_local.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack_local.extensions.secondorder.diag_ggn.convtransposend import (
    BatchDiagGGNConvTransposeND,
    DiagGGNConvTransposeND,
)


class DiagGGNConvTranspose2d(DiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )


class BatchDiagGGNConvTranspose2d(BatchDiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
