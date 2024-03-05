"""Partial derivatives for the Tanh activation function."""

from torch import Tensor
from torch.nn import Tanh

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class TanhDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        return False

    def df(
        self,
        module: Tanh,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ) -> Tensor:
        output = subsample(module.output, subsampling=subsampling)
        return 1.0 - output**2

    def d2f(self, module, g_inp, g_out):
        return -2.0 * module.output * (1.0 - module.output**2)
