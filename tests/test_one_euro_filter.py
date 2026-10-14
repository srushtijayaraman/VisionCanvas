"""Unit tests for OneEuroFilter."""

import pytest
from src.utils.one_euro_filter import OneEuroFilter


def test_one_euro_filter_initialization():
    f = OneEuroFilter(t0=0.0, x0=100.0, mincutoff=1.0, beta=0.01)
    assert f.x_prev == 100.0
    assert f.t_prev == 0.0


def test_one_euro_filter_zero_or_negative_dt():
    """Ensure filter does not divide by zero when called with same or earlier timestamp."""
    f = OneEuroFilter(t0=1.0, x0=100.0)
    # Same timestamp
    out1 = f(1.0, 105.0)
    assert out1 == 100.0
    # Slightly earlier timestamp
    out2 = f(0.9, 110.0)
    assert out2 == 100.0


def test_one_euro_filter_stationary_noise():
    """Stationary input with noise should be smoothed."""
    f = OneEuroFilter(t0=0.0, x0=100.0, mincutoff=1.0, beta=0.0)
    results = []
    for i in range(1, 20):
        t = i * 0.033
        # Jitter around 100.0
        val = 100.0 + (1.0 if i % 2 == 0 else -1.0)
        smoothed = f(t, val)
        results.append(smoothed)

    # Filtered output variation should be significantly smaller than raw noise
    diffs = [abs(r - 100.0) for r in results]
    assert all(d < 0.8 for d in diffs[5:])


def test_one_euro_filter_reset():
    """Reset should reinitialize filter state."""
    f = OneEuroFilter(t0=0.0, x0=100.0)
    f(0.1, 150.0)
    assert f.x_prev != 100.0

    f.reset(t0=1.0, x0=500.0)
    assert f.x_prev == 500.0
    assert f.t_prev == 1.0
    assert f.dx_prev == 0.0
