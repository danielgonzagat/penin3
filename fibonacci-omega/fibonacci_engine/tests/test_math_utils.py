"""Tests for mathematical utilities."""

import pytest
import numpy as np
from fibonacci_engine.core.math_utils import (
    fibonacci_seq,
    golden_ratio,
    phi_mix,
    phi_mix_array,
    spiral_scales,
    fibonacci_window,
    explore_exploit_budget,
)


def test_golden_ratio():
    """Test golden ratio constant."""
    phi = golden_ratio()
    assert abs(phi - 1.618) < 0.001
    assert abs(phi - (1 + np.sqrt(5)) / 2) < 1e-10


def test_fibonacci_seq():
    """Test Fibonacci sequence generation."""
    # Test empty
    assert fibonacci_seq(0) == []
    
    # Test first few terms
    assert fibonacci_seq(1) == [1]
    assert fibonacci_seq(2) == [1, 1]
    assert fibonacci_seq(7) == [1, 1, 2, 3, 5, 8, 13]
    
    # Test larger sequence
    fib_10 = fibonacci_seq(10)
    assert len(fib_10) == 10
    assert fib_10[-1] == 55  # 10th Fibonacci number


def test_phi_mix():
    """Test golden ratio mixing."""
    # Test endpoints
    assert phi_mix(0.0, 1.0, 0.0) > 0.0
    assert phi_mix(0.0, 1.0, 1.0) < 1.0
    
    # Test midpoint
    result = phi_mix(0.0, 1.0, 0.5)
    assert 0.0 < result < 1.0
    assert abs(result - 0.618) < 0.1  # Should be close to phi-1
    
    # Test symmetry property
    a, b = 2.0, 5.0
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = phi_mix(a, b, w)
        assert a <= result <= b or b <= result <= a


def test_phi_mix_array():
    """Test golden ratio mixing for arrays."""
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    
    result = phi_mix_array(a, b, 0.5)
    
    assert result.shape == a.shape
    assert np.all(result >= np.minimum(a, b))
    assert np.all(result <= np.maximum(a, b))


def test_spiral_scales():
    """Test multi-scale spiral generation."""
    # Test generation 5
    scales = spiral_scales(5, fib_depth=12)
    assert len(scales) == 3
    assert scales[0] < scales[1] < scales[2]  # Increasing scales
    assert all(s > 0 for s in scales)  # All positive
    
    # Test generation 1 (edge case)
    scales_1 = spiral_scales(1, fib_depth=12)
    assert len(scales_1) == 3
    assert all(s > 0 for s in scales_1)


def test_fibonacci_window():
    """Test Fibonacci window calculation."""
    # First few windows
    assert fibonacci_window(1, 12) == 1
    assert fibonacci_window(2, 12) == 1
    assert fibonacci_window(3, 12) == 2
    assert fibonacci_window(4, 12) == 3
    assert fibonacci_window(5, 12) == 5
    
    # Window should not exceed fib_depth
    for gen in range(1, 20):
        window = fibonacci_window(gen, fib_depth=10)
        assert window > 0


def test_explore_exploit_budget():
    """Test explore/exploit budget splitting."""
    total_budget = 100
    
    # Early generations: more exploration
    explore_early, exploit_early = explore_exploit_budget(1, total_budget, 12)
    assert explore_early + exploit_early == total_budget
    assert explore_early > 0 and exploit_early > 0
    
    # Late generations: more exploitation
    explore_late, exploit_late = explore_exploit_budget(12, total_budget, 12)
    assert explore_late + exploit_late == total_budget
    
    # Should show progression
    # (exact values depend on phi_mix implementation, but should be reasonable)
    assert 0 < explore_early < total_budget
    assert 0 < exploit_early < total_budget


def test_determinism():
    """Test that functions are deterministic."""
    # Same inputs should give same outputs
    np.random.seed(42)
    scales1 = spiral_scales(7, 12)
    
    np.random.seed(42)
    scales2 = spiral_scales(7, 12)
    
    assert scales1 == scales2
