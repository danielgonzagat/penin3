import random
from darwinacci_omega.core.evaluator import EvaluatorPipeline
from darwinacci_omega.core.env_plugins import REGISTRY


def test_portfolio_dummy_symbolic():
    base = lambda g, r: {'objective': 0.0, 'behavior': [0.0, 0.0]}
    portfolio = [REGISTRY.get('dummy_symbolic')]
    pipe = EvaluatorPipeline(base=base, portfolio=portfolio)
    pipe.use_portfolio = True
    out = pipe.evaluate({'a': 1.0, 'b': 2.0}, random.Random(7))
    assert 'objective' in out and 'behavior' in out
