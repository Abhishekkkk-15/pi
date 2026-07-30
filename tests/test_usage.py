from config import estimate_cost


def test_estimate_cost():
    assert estimate_cost(1_000_000, 1_000_000, 0.50, 1.50) == 2.0
    assert abs(estimate_cost(10200, 2250, 0.50, 1.50) - 0.008475) < 1e-9


def test_estimate_cost_zero():
    assert estimate_cost(0, 0, 0.50, 1.50) == 0.0

