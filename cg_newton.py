# python3.11.6
"""Implementation of the Conjugate Gradient Optimiser."""

import torch
import math
from collections.abc import Iterable, Callable


class CGN(torch.optim.Optimizer):
    """
    Implement the Conjugate Gradient Optimizer as a PyTorch optimizer.

    This optimizer adjusts the parameters based on the conjugate gradient method
    applied to the gradients of the parameters.

    Attributes
    ----------
        parameters (Iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        damping (float): Damping term to improve convergence properties.
        maxiter (int): Maximum number of iterations for the conjugate gradient method.
        tol (float): Tolerance for convergence.
        atol (float): Absolute tolerance for convergence.
    """

    def __init__(
        self,
        parameters: Iterable,
        lr: float = 0.1,
        damping: float = 1e-2,
        maxiter: int = 100,
        tol: float = 1e-1,
        atol: float = 1e-8,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": lr,
                "damping": damping,
                "maxiter": maxiter,
                "tol": tol,
                "atol": atol,
            },
        )

    def step(self) -> None:
        """
        Perform a single optimization step.

        This method iterates over each parameter group and its parameters,
        computes the direction for parameter updates using the conjugate gradient
        method and updates the parameters accordingly.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                damped_curvature = self.damped_matvec(p, group["damping"])

                direction, _ = self.cg(
                    damped_curvature,
                    -p.grad.data,
                    maxiter=group["maxiter"],
                    tol=group["tol"],
                    atol=group["atol"],
                )

                p.data.add_(direction, alpha=group["lr"])

    def damped_matvec(self, param: torch.Tensor, damping: float) -> torch.Tensor:
        """
        Return a function that computes the damped matrix-vector product.

        The damping improves the conditioning of the problem, which can
        accelerate convergence in the conjugate gradient method.

        Args:
            param (torch.Tensor): The parameter tensor.
            damping (float): The damping coefficient.

        Returns
        -------
            A function that takes a vector `v` and returns the damped matrix-vector
            product with the Hessian of the loss function with respect to `param`.
        """

        def matvec(v: torch.Tensor) -> torch.Tensor:
            v = v.view_as(param)
            hv = self.hessian_vector_product(param, v)
            return damping * v + hv

        return matvec

    def hessian_vector_product(
        self, param: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Hessian-vector product for a given parameter and vector.

        This method is used to efficiently compute the curvature information
        without explicitly forming the Hessian matrix.

        Args:
            param (torch.Tensor): The parameter tensor.
            v (torch.Tensor): The vector to be multiplied with the Hessian.

        Returns
        -------
            The Hessian-vector product as a torch.Tensor.
        """
        if param.grad is None or not param.grad.requires_grad:
            return torch.zeros_like(v)

        grad_param = torch.autograd.grad(
            outputs=param.grad,
            inputs=param,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True,
        )[0]
        return grad_param

    @staticmethod
    def cg(
        a: Callable,
        b: torch.Tensor,
        x0: torch.Tensor = None,
        maxiter: int = -1,
        tol: float = 1e-5,
        atol: float = 1e-8,
    ) -> tuple[torch.Tensor, int]:
        r"""Solve :math:`ax = b` for :math:`x` using conjugate gradient.

        The interface is similar to CG provided by :code:`scipy.sparse.linalg.cg`.

        The main iteration loop follows the pseudo code from Wikipedia:
            https://en.wikipedia.org/w/index.php?title=Conjugate_gradient_method&oldid=855450922

        Parameters
        ----------
        a : function
            Function implementing matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system.
        x0 : torch.Tensor
            Initialization estimate.
        atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `atol`
        tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `tol`.
        maxiter: int
            Maximum number of iterations.

        Returns
        -------
        x (torch.Tensor): Approximate solution :math:`x` of the linear system
        info (int): Provides convergence information, if CG converges info
                    corresponds to numiter, otherwise info is set to zero.
        """
        maxiter = b.numel() if maxiter == -1 else min(maxiter, b.numel())
        x = torch.zeros_like(b) if x0 is None else x0

        # initialize parameters
        r = (b - a(x)).detach()
        p = r.clone()
        rs_old = (r**2).sum().item()

        # stopping criterion
        norm_bound = max([tol * torch.norm(b).item(), atol])

        def converged(rs: float, numiter: int) -> tuple[bool, int]:
            """Check whether CG stops (convergence or steps exceeded)."""
            norm_converged = norm_bound > math.sqrt(rs)
            info = numiter if norm_converged else 0
            iters_exceeded = numiter > maxiter
            return (norm_converged or iters_exceeded), info

        # iterate
        iterations = 0
        while True:
            ap = a(p).detach()

            alpha = rs_old / (p * ap).sum().item()
            x.add_(p, alpha=alpha)
            r.sub_(ap, alpha=alpha)
            rs_new = (r**2).sum().item()
            iterations += 1

            stop, info = converged(rs_new, iterations)
            if stop:
                return x, info

            p.mul_(rs_new / rs_old)
            p.add_(r)
            rs_old = rs_new
