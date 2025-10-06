import math
from darwinacci_omega.core.f_clock import TimeCrystal
from darwinacci_omega.core.golden_spiral import GoldenSpiralArchive


def test_time_crystal_budget_bounds():
    tc = TimeCrystal(max_cycles=5)
    for c in range(1, 6):
        b = tc.budget(c)
        assert 6 <= b.generations <= 96
        assert 0.02 <= b.mut <= 0.45
        assert 0.20 <= b.cx <= 0.95
        assert b.elite >= 2
        assert isinstance(b.checkpoint, bool)


def test_golden_spiral_coverage_monotonic():
    arch = GoldenSpiralArchive(bins=8)
    coverages = []
    # Use evenly spaced angles to ensure different bins
    for i in range(8):
        theta = 2 * math.pi * i / 8.0
        behavior = [math.cos(theta), math.sin(theta)]
        arch.add(behavior, score=float(i))
        coverages.append(arch.coverage())
    # Coverage should be non-decreasing and reach at least half
    assert all(coverages[i] <= coverages[i+1] for i in range(len(coverages)-1))
    assert coverages[-1] >= 0.5
