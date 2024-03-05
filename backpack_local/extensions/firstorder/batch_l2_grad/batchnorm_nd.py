"""Contains batch_l2 extension for BatchNorm."""

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack_local.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack_local.extensions.backprop_extension import BackpropExtension
from backpack_local.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base
from backpack_local.utils.errors import batch_norm_raise_error_if_train


class BatchL2BatchNorm(BatchL2Base):
    """batch_l2 extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], BatchNormNdDerivatives())

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: BatchNorm1d | BatchNorm2d | BatchNorm3d,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
    ) -> None:
        batch_norm_raise_error_if_train(module)
