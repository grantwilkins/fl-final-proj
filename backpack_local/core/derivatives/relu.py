"""Partial derivatives for the ReLU activation function."""

from torch import Tensor, gt
from torch.nn import ReLU

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class ReLUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        """`ReLU''(x) = 0`."""
        return True

    def df(
        self,
        module: ReLU,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ) -> Tensor:
        """First ReLU derivative: `ReLU'(x) = 0 if x < 0 else 1`."""
        input0 = subsample(module.input0, subsampling=subsampling)
        return gt(input0, 0).to(input0.dtype)
