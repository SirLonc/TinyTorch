"""Numerical checks for the scalar sigmoid and its local backward callback."""

import math
from decimal import Decimal, localcontext

import pytest

from tinytorch.autograd import Value


@pytest.mark.parametrize("data", [-1000.0, -40.0, -2.0, 0.0, 2.0, 40.0, 1000.0])
def test_sigmoid_matches_high_precision_reference(data: float) -> None:
    with localcontext() as context:
        context.prec = 80
        expected = float(1 / (1 + (-Decimal.from_float(data)).exp()))

    result = Value(data).sigmoid()

    assert math.isfinite(result.data)
    assert 0 <= result.data <= 1
    assert result.data == pytest.approx(expected, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("data", [-40.0, -2.0, 0.0, 2.0, 40.0])
def test_sigmoid_gradient_matches_finite_difference(data: float) -> None:
    # Compute finite differences in Decimal so saturation in float arithmetic
    # does not erase the small, nonzero derivative at x=40.
    with localcontext() as context:
        context.prec = 80
        point = Decimal.from_float(data)
        step = Decimal("1e-6")
        upper = 1 / (1 + (-(point + step)).exp())
        lower = 1 / (1 + (-(point - step)).exp())
        expected = float((upper - lower) / (2 * step))

    value = Value(data)
    result = value.sigmoid()
    result.grad = 3.0
    result._backward()

    assert value.grad == pytest.approx(3 * expected, rel=1e-9, abs=0.0)


def test_sigmoid_backward_accumulates_gradient() -> None:
    value = Value(0.0)
    value.grad = 2.0
    result = value.sigmoid()
    result.grad = 4.0

    result._backward()
    result._backward()

    assert value.grad == 4.0


@pytest.mark.parametrize("data", [-1000.0, 1000.0])
def test_sigmoid_extreme_input_has_finite_gradient(data: float) -> None:
    value = Value(data)
    result = value.sigmoid()
    result.grad = 3.0

    result._backward()

    assert math.isfinite(value.grad)
    assert value.grad == 0.0
