from backpack_local.core.derivatives.conv1d import Conv1DDerivatives
from backpack_local.extensions.secondorder.diag_ggn.convnd import (
    BatchDiagGGNConvND,
    DiagGGNConvND,
)


class DiagGGNConv1d(DiagGGNConvND):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])


class BatchDiagGGNConv1d(BatchDiagGGNConvND):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
