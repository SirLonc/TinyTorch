"""Scalar operations, graph construction and backpropagation (stage 1)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import math


class Value:
    """A scalar value, its gradient and connections to its input values."""

    def __init__(
        self,
        data: float,
        children: Iterable[Value] = (),
        operator: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self.prev = set(children)
        self.operator = operator
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other_value: Value | float | int) -> Value:
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        out = Value(self.data + other_value.data, (self, other_value), operator="+")

        def backward() -> None:
            self.grad += out.grad
            other_value.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other_value: Value | float | int) -> Value:
        return self + other_value

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other_value: Value | float | int) -> Value:
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        return self + (-other_value)

    def __rsub__(self, other_value: Value | float | int) -> Value:
        """TODO: support expressions such as 2 - Value(3)."""
        return (-self) + other_value

    def __mul__(self, other_value: Value | float | int) -> Value:
        other_value = other_value if isinstance(other_value, Value) else Value(other_value)
        out = Value(self.data * other_value.data, (self, other_value), operator="*")

        def backward() -> None:
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other_value: Value | float | int) -> Value:
        return self * other_value

    def __truediv__(self, other_value: Value | float | int) -> Value:
        return self * other_value**-1

    def __rtruediv__(self, other_value: Value | float | int) -> Value:
        """TODO: support expressions such as 2 / Value(3) using existing operations."""
        return other_value * self**-1

    def __pow__(self, exponent: float | int) -> Value:
        """TODO: compute a constant power and register its local backward callback."""
        out = Value(self.data ** exponent, (self,), operator="**")

        def backward() -> None:
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent-1) * out.grad

        out._backward = backward
        return out

    def exp(self) -> Value:
        """TODO: compute the exponential and register its local backward callback."""
        out = Value(math.exp(self.data), (self,), operator="exp")

        def backward() -> None:
            self.grad += math.exp(self.data) * out.grad

        out._backward = backward
        return out

    def log(self) -> Value:
        """TODO: compute the natural logarithm for positive data and its gradient."""
        out = Value(math.log(self.data), (self,), operator='log')

        def backward() -> None:
            self.grad += 1/self.data * out.grad

        out._backward = backward
        return out

    def relu(self) -> Value:
        """TODO: implement ReLU; use a zero derivative at the origin."""
        out = Value(self.data if self.data > 0 else 0, (self,), operator='relu')

        def backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = backward
        return out

    def tanh(self) -> Value:
        """TODO: compute the hyperbolic tangent and its local gradient."""
        tanh = math.tanh(self.data)
        out = Value(tanh, (self,), operator='tanh')

        def backward() -> None:
            self.grad += (1 - tanh**2) * out.grad

        out._backward = backward
        return out

    def sigmoid(self) -> Value:
        """Compute sigmoid without exponentiating a large positive number."""
        if self.data >= 0:
            exp_value = math.exp(-self.data)
            sigmoid = 1 / (1 + exp_value)
        else:
            # Multiply numerator and denominator by exp(x):
            # 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x)).
            exp_value = math.exp(self.data)
            sigmoid = exp_value / (1 + exp_value)

        out = Value(sigmoid, (self,), operator="sigmoid")

        def backward() -> None:
            # Equivalent to sigmoid * (1 - sigmoid), but avoids cancellation
            # when the computed sigmoid has rounded to exactly 1.0.
            self.grad += exp_value / (1 + exp_value) ** 2 * out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        """Propagate gradients through the graph ending at this value.

        TODO:
        1. Build a topological ordering through prev, visiting each node once.
        2. Seed the output gradient with 1.0.
        3. Call each node's _backward in reverse topological order.

        _backward handles one operation; backward traverses the entire graph.
        """
        if not self.prev:
            self.grad += 1
            return
        stack = []
        visited = set()
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    dfs(child)
                stack.append(v)
        dfs(self)

        self.grad = 1.0
        for node in reversed(stack):
            node._backward()

        for node in reversed(stack):
            if node.prev:
                node.grad = 0.0
