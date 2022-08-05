import torch
from torch import nn
from torch.distributions import transforms
from torch.distributions.constraint_registry import (
    ConstraintRegistry,
    biject_to,
    constraints,
    transform_to,
)
from torch.distributions.constraints import greater_than_eq, positive, simplex

torch_biject_to = biject_to
torch_transform_to = transform_to

# custom registry for NF transformations
nf_constraints_standard = ConstraintRegistry()


class SoftplusTransform(transforms.Transform):
    r"""
    Transform via the mapping :math:`y = \log(1 + \exp(x))`
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __init__(self, threshold: int = 20, beta: float = 1) -> None:
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.softplus = nn.Softplus(beta=self.beta, threshold=self.threshold)
        self.log_sigmoid = nn.LogSigmoid()

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(
            y * self.beta > self.threshold, y, (self.beta * y).expm1().log() / self.beta
        )

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_sigmoid(self.beta * x)


class LogSoftmaxTransform(transforms.Transform):
    r"""
    not bijective
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, cache_size=0):
        super().__init__(cache_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def __eq__(self, other):
        return isinstance(other, LogSoftmaxTransform)

    def _call(self, x):
        return self.logsoftmax(x)


@nf_constraints_standard.register(positive)
def _transform_to_positive_soft(constraint):
    return SoftplusTransform()


@nf_constraints_standard.register(simplex)
def _transform_to_simplex_log(constraint):
    return LogSoftmaxTransform()


@nf_constraints_standard.register(greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform(
        [SoftplusTransform(), transforms.AffineTransform(constraint.lower_bound, 1)]
    )


# Alternative versions (eg Exp vs Softplus)
nf_constraints_alternative = ConstraintRegistry()


@nf_constraints_alternative.register(positive)
def _transform_to_positive(constraint):
    return transforms.ExpTransform()

@nf_constraints_alternative.register(greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform(
        [transforms.ExpTransform(), transforms.AffineTransform(constraint.lower_bound, 1)]
    )