# python3.11.6
"""Implementation of the Gauss Newton Optimiser."""

import torch
from collections.abc import Iterable
import numpy as np


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
                if len(p.kflr) == 2:  # noqa:PLR2004
                    q, g = p.kflr
                    k = group["damping"]
                    iq = torch.eye(q.shape[0], device=q.device)
                    ig = torch.eye(g.shape[0], device=g.device)
                    # w = torch.sqrt(
                    #     torch.norm(q.cpu(), p="nuc") / torch.norm(g.cpu(), p="nuc")
                    # ).to(device="mps")
                    # print(w)
                    w = 0.1
                    orig_shape = p.grad.shape
                    left = torch.inverse(q + (w * np.sqrt(k) + iq))
                    right = torch.inverse((g + (w**-1 * np.sqrt(k) * ig)).cpu()).to(
                        device="mps"
                    )
                    if len(p.grad.shape) == 4:  # noqa: PLR2004
                        step_direction = left @ p.grad.flatten(1) @ right
                    elif len(p.grad.shape) in {1, 2}:
                        step_direction = left @ p.grad @ right
                    else:
                        # print("aaaa")
                        # print(p.grad.shape)
                        # exit()
                        raise ValueError()
                    p.data.add_(
                        torch.reshape(step_direction, orig_shape),
                        alpha=-group["step_size"],
                    )
                elif len(p.kflr) == 1:
                    c = p.kflr[0]
                    k = group["damping"]
                    step_direction = (
                        torch.inverse(c + (k * torch.eye(c.shape[0], device=c.device)))
                        @ p.grad
                    )
                    p.data.add_(
                        step_direction,
                        alpha=-group["step_size"],
                    )
                else:
                    # should only have one or two factors
                    # print([i.shape for i in p.kflr])
                    # print("aaa2")
                    raise ValueError()
