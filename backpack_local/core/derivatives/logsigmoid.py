"""Contains partial derivatives for the ``torch.nn.LogSigmoid`` layer."""

from torch import Tensor, exp
from torch.nn import LogSigmoid

from backpack_local.core.derivatives.elementwise import ElementwiseDerivatives
from backpack_local.utils.subsampling import subsample


class LogSigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        """`logsigmoid''(x) ≠ 0`."""
        return False

    def df(
        self,
        module: LogSigmoid,
        g_inp: tuple[Tensor],
        g_out: tuple[Tensor],
        subsampling: list[int] = None,
    ) -> Tensor:
        """First Logsigmoid derivative: `logsigmoid'(x) = 1 / (e^x + 1) `."""
        input0 = subsample(module.input0, subsampling=subsampling)
        return 1 / (exp(input0) + 1)

    def d2f(self, module, g_inp, g_out):
        """Second Logsigmoid derivative: `logsigmoid''(x) = - e^x / (e^x + 1)^2`."""
        exp_input = exp(module.input0)
        return -(exp_input / (exp_input + 1) ** 2)
