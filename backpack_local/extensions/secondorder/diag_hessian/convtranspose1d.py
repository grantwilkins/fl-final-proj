from backpack_local.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack_local.extensions.secondorder.diag_hessian.convtransposend import (
    BatchDiagHConvTransposeND,
    DiagHConvTransposeND,
)


class DiagHConvTranspose1d(DiagHConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )


class BatchDiagHConvTranspose1d(BatchDiagHConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
