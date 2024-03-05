from backpack_local.core.derivatives.conv2d import Conv2DDerivatives
from backpack_local.extensions.secondorder.diag_ggn.convnd import (
    BatchDiagGGNConvND,
    DiagGGNConvND,
)


class DiagGGNConv2d(DiagGGNConvND):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])


class BatchDiagGGNConv2d(BatchDiagGGNConvND):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
