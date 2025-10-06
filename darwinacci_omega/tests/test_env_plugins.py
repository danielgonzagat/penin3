import random
from darwinacci_omega.core.env_plugins import REGISTRY, register_gym_env


def test_register_dummy_symbolic_present():
    assert 'dummy_symbolic' in REGISTRY.list()


def test_register_gym_env_smoke():
    ok = register_gym_env('CartPole-v1')
    # ok is False if gym unavailable in env; both are acceptable for CI
    if ok:
        fn = REGISTRY.get('gym::CartPole-v1')
        out = fn({'w': 0.1, 'b': 0.0}, random.Random(7))
        assert 'objective' in out
