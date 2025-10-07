from omega_ext.core.bridge import DarwinOmegaBridge
from omega_ext.plugins.adapter_darwin import autodetect
def test_quick():
    init_fn, eval_fn = autodetect()
    eng = DarwinOmegaBridge(init_fn, eval_fn, seed=123, max_cycles=3)
    champ = eng.run(max_cycles=3)
    assert champ is not None and champ.score >= 0.0
    print("[OK] Omega bridge champion:", champ.score)

if __name__ == "__main__":
    test_quick()
