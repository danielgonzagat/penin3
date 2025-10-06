import os
from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.plugins import toy


def test_seed_determinism_short():
    os.environ['DARWINACCI_TRIALS'] = '1'
    e1 = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=2, pop_size=8, seed=123)
    e2 = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=2, pop_size=8, seed=123)
    c1 = e1.run(max_cycles=2)
    c2 = e2.run(max_cycles=2)
    assert round(c1.score, 6) == round(c2.score, 6)


def test_long_run_smoke():
    # Keep small to fit CI wallclock; just ensures no exceptions
    os.environ['DARWINACCI_TRIALS'] = '1'
    e = DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=3, pop_size=16, seed=7)
    c = e.run(max_cycles=3)
    assert c is not None
