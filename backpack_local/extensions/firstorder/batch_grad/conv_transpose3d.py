from backpack_local.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack_local.extensions.firstorder.batch_grad.batch_grad_base import (
    BatchGradBase,
)


class BatchGradConvTranspose3d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )
