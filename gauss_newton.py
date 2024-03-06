# python3.11.6
"""Implementation of the Gauss Newton Optimiser."""

import torch
from collections.abc import Iterable


class DGN(torch.optim.Optimizer):
    """Diagonal Gauss Newton Torch Optimiser."""

    def __init__(self, parameters: Iterable, step_size: float, damping: float) -> None:
        super().__init__(parameters, {"step_size": step_size, "damping": damping})

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                # print(f"gf: {p.diag_ggn_exact.shape}")
                # print(f"grad: {p.grad.shape}")
                step_direction = p.grad / (p.diag_ggn_exact + group["damping"])
                p.data.add_(step_direction, alpha=-group["step_size"])


class BDGN(torch.optim.Optimizer):
    """Block Diagonal Gauss Newton Torch Optimiser."""

    def __init__(self, parameters: Iterable, step_size: float, damping: float) -> None:
        super().__init__(parameters, {"step_size": step_size, "damping": damping})

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                print([i.shape for i in p.kflr])  # noqa:T201
                # if len(p.kflr) > 1:
                #     if len(p.kflr) == 2:
                #         q, g = p.kflr
                #     elif len(p.kflr) == 4:
                #         q = torch.kron(p.kflr[0], p.kflr[1])  # p.kflr[0]  #
                #         g = torch.kron(p.kflr[2], p.kflr[3])  # p.kflr[2]  #
                #     k = group["damping"]
                #     w = 1
                #     orig_shape = p.grad.shape
                #     left = torch.inverse(
                #         q + (w * np.sqrt(k) + torch.eye(q.shape[0], device=q.device))
                #     )
                #     right = torch.inverse(
                #         (
                #             g
                #             + (
                #                 w**-1
                #                 * np.sqrt(k)
                #                 * torch.eye(g.shape[0], device=g.device)
                #             )
                #         ).cpu()
                #     ).to(device="mps")
                #     step_direction = left @ p.grad.flatten(1) @ right
                #     p.data.add_(
                #         torch.reshape(step_direction, orig_shape),
                #         alpha=-group["step_size"],
                #     )
                # elif len(p.kflr) == 1:
                #     c = p.kflr[0]
                #     k = group["damping"]
                #     step_direction = (
                #         torch.inverse(c +
                #  (k * torch.eye(c.shape[0], device=c.device)))
                #         @ p.grad
                #     )
                #     p.data.add_(
                #         step_direction,
                #         alpha=-group["step_size"],
                #     )
                # else:
                #     print(len(p.kflr))
                #     print(p.kflr)
                #     for item in p.kflr:
                #         print(item.shape)
                #     print("hmm")
                #     exit()
