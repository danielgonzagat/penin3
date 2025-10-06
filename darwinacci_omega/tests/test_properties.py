from hypothesis import given, strategies as st, settings, HealthCheck
from darwinacci_omega.core.darwin_ops import prune_genes
from darwinacci_omega.core.novelty_phi import Novelty


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    g=st.dictionaries(
        keys=st.text(min_size=1, max_size=6),
        values=st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=0, max_size=64
    ),
    max_genes=st.integers(min_value=1, max_value=128)
)
def test_prune_genes_size_and_keys(g, max_genes):
    from copy import deepcopy
    g0 = deepcopy(g)
    out = prune_genes(g, max_genes=max_genes, rng=__import__('random').Random(123))
    assert len(out) <= max_genes
    # No new keys introduced
    assert set(out.keys()).issubset(set(g0.keys()))


@given(
    b=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=5)
)
def test_novelty_non_increase_after_self_add(b):
    n = Novelty(k=3, max_size=1000)
    before = n.score(b)
    n.add(b)
    after = n.score(b)
    assert after <= before
