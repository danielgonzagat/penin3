"""
Mathematical utilities for Fibonacci Engine.

This module implements the core mathematical operations:
- Fibonacci sequence generation
- Golden ratio (Φ) calculations
- Phi mixing for balanced exploration/exploitation
- Multi-scale spiral generation
"""

import math
from typing import List, Tuple
import numpy as np


# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618


def golden_ratio() -> float:
    """
    Return the golden ratio Φ = (1 + √5) / 2 ≈ 1.618.
    
    Returns:
        float: The golden ratio constant.
    """
    return PHI


def fibonacci_seq(n: int) -> List[int]:
    """
    Generate the Fibonacci sequence up to n terms.
    
    F(1) = 1, F(2) = 1, F(k) = F(k-1) + F(k-2)
    
    Args:
        n: Number of terms to generate.
        
    Returns:
        List of first n Fibonacci numbers.
        
    Example:
        >>> fibonacci_seq(7)
        [1, 1, 2, 3, 5, 8, 13]
    """
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]
    
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib


def phi_mix(a: float, b: float, w: float = 0.5) -> float:
    """
    Mix two values using golden ratio weighting.
    
    The mixing coefficient α is computed as:
        α = (Φ - 1) * w + (2 - Φ) * (1 - w)
    
    Then: result = (1 - α) * a + α * b
    
    Args:
        a: First value.
        b: Second value.
        w: Weight parameter in [0, 1]. Default 0.5.
        
    Returns:
        Mixed value with golden ratio proportions.
        
    Example:
        >>> phi_mix(0.0, 1.0, 0.5)
        0.618...
    """
    w = max(0.0, min(1.0, w))  # Clamp to [0, 1]
    alpha = (PHI - 1) * w + (2 - PHI) * (1 - w)
    return (1 - alpha) * a + alpha * b


def phi_mix_array(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    """
    Mix two numpy arrays using golden ratio weighting.
    
    Args:
        a: First array.
        b: Second array (must be same shape as a).
        w: Weight parameter in [0, 1]. Default 0.5.
        
    Returns:
        Mixed array with golden ratio proportions.
    """
    w = max(0.0, min(1.0, w))
    alpha = (PHI - 1) * w + (2 - PHI) * (1 - w)
    return (1 - alpha) * a + alpha * b


def spiral_scales(generation: int, fib_depth: int = 12) -> Tuple[float, float, float]:
    """
    Generate three perturbation scales for multi-scale spiral search.
    
    Scales are proportional to F(g-2), F(g-1), F(g) normalized by the
    current window size.
    
    Args:
        generation: Current generation number (1-indexed).
        fib_depth: Maximum depth for Fibonacci sequence.
        
    Returns:
        Tuple of three scales (small, medium, large).
        
    Example:
        >>> spiral_scales(5, 12)
        (0.125, 0.25, 0.375)
    """
    fib = fibonacci_seq(fib_depth)
    
    # Clamp generation to valid range
    gen_idx = min(generation, len(fib)) - 1
    gen_idx = max(0, gen_idx)
    
    # Get F(g-2), F(g-1), F(g) with bounds checking
    if gen_idx >= 2:
        f_g_minus_2 = fib[gen_idx - 2]
        f_g_minus_1 = fib[gen_idx - 1]
        f_g = fib[gen_idx]
    elif gen_idx == 1:
        f_g_minus_2 = 1
        f_g_minus_1 = fib[0]
        f_g = fib[1]
    else:  # gen_idx == 0
        f_g_minus_2 = 1
        f_g_minus_1 = 1
        f_g = fib[0]
    
    # Normalize by current window
    window = fib[gen_idx]
    
    scale_small = f_g_minus_2 / (window * 8)
    scale_medium = f_g_minus_1 / (window * 4)
    scale_large = f_g / (window * 2)
    
    return (scale_small, scale_medium, scale_large)


def fibonacci_window(generation: int, fib_depth: int = 12) -> int:
    """
    Get the Fibonacci window size for a given generation.
    
    Args:
        generation: Current generation number (1-indexed).
        fib_depth: Maximum depth for Fibonacci sequence.
        
    Returns:
        Window size (number of evaluations/samples).
    """
    fib = fibonacci_seq(fib_depth)
    idx = min(generation - 1, len(fib) - 1)
    return fib[idx]


def explore_exploit_budget(generation: int, total_budget: int, 
                           fib_depth: int = 12) -> Tuple[int, int]:
    """
    Split budget between exploration and exploitation using Φ.
    
    As generations progress, gradually shift from exploration to exploitation
    using golden ratio proportions.
    
    Args:
        generation: Current generation number.
        total_budget: Total computational budget.
        fib_depth: Maximum depth for Fibonacci sequence.
        
    Returns:
        Tuple of (explore_budget, exploit_budget).
    """
    # Progress from 0 (early) to 1 (late)
    progress = min(generation / fib_depth, 1.0)
    
    # Early: more exploration; Late: more exploitation
    # Use phi_mix to smoothly transition
    exploit_ratio = phi_mix(0.382, 0.618, progress)  # From ~0.4 to ~0.6
    
    exploit_budget = int(total_budget * exploit_ratio)
    explore_budget = total_budget - exploit_budget
    
    return (explore_budget, exploit_budget)
