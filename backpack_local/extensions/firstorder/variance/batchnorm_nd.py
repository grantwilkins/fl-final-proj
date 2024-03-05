"""Variance extension for BatchNorm."""

from torch import Tensor
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from backpack_local.extensions.backprop_extension import BackpropExtension
from backpack_local.extensions.firstorder.gradient.batchnorm_nd import GradBatchNormNd
from backpack_local.extensions.firstorder.sum_grad_squared.batchnorm_nd import (
    SGSBatchNormNd,
)
from backpack_local.extensions.firstorder.variance.variance_base import (
    VarianceBaseModule,
)
from backpack_local.utils.errors import batch_norm_raise_error_if_train


class VarianceBatchNormNd(VarianceBaseModule):
    """Variance extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], GradBatchNormNd(), SGSBatchNormNd())

    def check_hyperparameters_module_extension(
        self,
        ext: BackpropExtension,
        module: BatchNorm1d | BatchNorm2d | BatchNorm3d,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
    ) -> None:
        batch_norm_raise_error_if_train(module)
