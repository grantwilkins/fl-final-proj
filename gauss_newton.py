# python3.11.6
"""Implementation of the Gauss Newton Optimiser."""

import torch
import numpy as np

# from distutils.version import LooseVersion

# if LooseVersion(torch.__version__) >= LooseVersion("2.0.0"):
#     pass
# else:
#     pass

# MIN_LEARNING_RATE = 0.0
# ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


# class GNA(torch.optim.Optimizer):
#     r"""Implements Gauss-Newton.

#     Args:
#         params (iterable): an iterable of :class:`torch.Tensor` s or
#             :class:`dict` s. Specifies what Tensors should be optimized.
#         lr: (float): learning rate
#         model: (nn.Module): NN.
#     """

#     def __init__(self, params: ParamsT, lr: float, model: nn.Module) -> None:
#         if lr is not None and lr < MIN_LEARNING_RATE:
#             raise ValueError(f"Invalid learning rate: {lr}")

#         defaults = {"lr": lr}

#         super().__init__(params, defaults)

#         self._model = model
#         self._params = self.param_groups[0]["params"]
#         self._j_list: list = []
#         self._h_list: list = []

#     def __setstate__(self, state: dict[str, Any]) -> None:
#         """Set optimizer state."""
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault("nesterov", False)
#             group.setdefault("amsgrad", False)

#     @torch.no_grad()
#     def step(
#         self, x: torch.Tensor, closure: Callable[[], float] | None = None
#     ) -> float | None:
#         """Perform a single optimization step.

#         Args:
#             x: Current data batch, which is needed to compute 2nd order derivatives
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params_with_grad = []
#             d_p_list = []
#             lr = group["lr"]

#             for p in group["params"]:
#                 if p.grad is not None:
#                     params_with_grad.append(p)
#                     d_p_list.append(p.grad)

#             parameters = dict(self._model.named_parameters())
#             keys, values = zip(*parameters.items(), strict=False)

#             self._h_list = []

#             # vectorized jacobian (https://github.com/pytorch/pytorch/issues/49171)
#             def func(*params: torch.Tensor) -> tuple[torch.Tensor] | torch.Tensor:
#                 out = torch.func.functional_call(
#                     self._model, dict(zip(keys, params, strict=False)), x
#                 )
#                 return out

#             # start = time.time()
#             # self._j_list: tuple[torch.Tensor] = torch.autograd.functional.jacobian(
#             #     func, values, create_graph=False
#             # )  # NxCxBxCxHxW
#             # end = time.time()
#             # print(f"jacobian autograd time: {end - start}")

#             start = time.time()
#             argsnums = tuple(range(len(values)))
#             self._j_list = jacobian(func, argnums=argsnums)(*values)
#             end = time.time()
#             print(f"jacobian reverse func time: {end - start}")

#             # comps = []
#             # for a, b in zip(self._j_list, jlist2):
#             #     comps.append(torch.allclose(a.flatten(), b.flatten(), atol=1e-04))
#             # print(f"equivalent: {all(comps)}")

#             # create hessian approximation
#             start = time.time()
#             for i, j in enumerate(self._j_list):
#                 j = j.flatten(
#                     end_dim=len(self._j_list[i].shape) - len(d_p_list[i].shape) - 1
#                 ).flatten(
#                     start_dim=1
#                 )  # (NC)x(BCHW)
#                 try:
#                     h = j.T.matmul(j)
#                 except RuntimeError:
#                     h = None
#                 self._h_list.append(h)
#             end = time.time()
#             print(f"hessian matrix mult time: {end - start}")

#             self.gna_update(
#                 params_with_grad,
#                 d_p_list,
#                 lr=lr,
#             )

#         return loss

#     def gna_update(
#         self,
#         params: list[torch.Tensor],
#         d_p_list: list[torch.Tensor],
#         lr: float,
#     ) -> None:
#         r"""Functional API that performs Gauss-Newton algorithm computation."""
#         assert len(d_p_list) == len(self._h_list), "Layer number mismatch"

#         for i, param in enumerate(params):

#             d_p = d_p_list[i]
#             h = self._h_list[i]
#             if h is None:
#                 param.add_(d_p, alpha=-lr)
#                 break

#             diag_vec = h.diagonal() + torch.finfo(h.dtype).eps * 1
#             h.as_strided([h.size(0)], [h.size(0) + 1]).copy_(diag_vec)

#             # torch.linalg.pinv()
#             print(h.dtype)
#             h_i = torch.empty_like(h)
#             print(h_i.dtype)
#             torch.linalg.pinv(h, out=h_i)
#             # h.pinverse()
#             # h_i = (
#             #     h.to(device="cpu")
#             #     .pinverse()
#             #     .to(device=obtain_device(), dtype=torch.float32)
#             # )
#             if h_i.shape[-1] == d_p.flatten().shape[0]:
#                 d2_p = h_i.matmul(d_p.flatten()).reshape(d_p_list[i].shape)
#                 param.add_(d2_p, alpha=-lr)
#             else:
#                 raise TypeError("Tensor dimension mismatch")


class DGN(torch.optim.Optimizer):
    def __init__(self, parameters, step_size, damping):
        super().__init__(parameters, dict(step_size=step_size, damping=damping))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                # print(f"gf: {p.diag_ggn_exact.shape}")
                # print(f"grad: {p.grad.shape}")
                step_direction = p.grad / (p.diag_ggn_exact + group["damping"])
                p.data.add_(step_direction, alpha=-group["step_size"])


class BDGN(torch.optim.Optimizer):
    def __init__(self, parameters, step_size, damping):
        super().__init__(parameters, dict(step_size=step_size, damping=damping))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.kflr) == 2:
                    Q, G = p.kflr
                    k = group["damping"]
                    w = 1
                    orig_shape = p.grad.shape
                    left = torch.inverse(
                        Q + (w * np.sqrt(k) + torch.eye(Q.shape[0], device=Q.device))
                    )
                    right = torch.inverse(
                        (
                            G
                            + (
                                w**-1
                                * np.sqrt(k)
                                * torch.eye(G.shape[0], device=G.device)
                            )
                        ).cpu()
                    ).to(device="mps")
                    step_direction = left @ p.grad.flatten(1) @ right
                    p.data.add_(
                        torch.reshape(step_direction, orig_shape),
                        alpha=-group["step_size"],
                    )
                else:
                    C = p.kflr[0]
                    k = group["damping"]
                    step_direction = (
                        torch.inverse(C + (k * torch.eye(C.shape[0], device=C.device)))
                        @ p.grad
                    )
                    p.data.add_(
                        step_direction,
                        alpha=-group["step_size"],
                    )
