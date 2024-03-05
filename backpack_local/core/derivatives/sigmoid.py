"""Partial derivatives for the Sigmoid activation function."""

from torch import Tensor
from torch.nn import Sigmoid

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class SigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        """`σ''(x) ≠ 0`."""
        return False

    def df(
        self,
        module: Sigmoid,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ) -> Tensor:
        """First sigmoid derivative: `σ'(x) = σ(x) (1 - σ(x))`."""
        output = subsample(module.output, subsampling=subsampling)
        return output * (1.0 - output)

    def d2f(self, module, g_inp, g_out):
        """Second sigmoid derivative: `σ''(x) = σ(x) (1 - σ(x)) (1 - 2 σ(x))`."""
        return module.output * (1 - module.output) * (1 - 2 * module.output)
