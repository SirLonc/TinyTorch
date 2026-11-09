"""Tensor storage, shape operations and differentiable arithmetic (stages 2–3)."""


class Tensor:
    """Placeholder for the multidimensional array and its computation graph."""

    def __init__(self, data: object, *, requires_grad: bool = False) -> None:
        raise NotImplementedError("Tensor will be implemented in roadmap stages 2–3.")
