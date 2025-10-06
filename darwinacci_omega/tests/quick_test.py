from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.plugins import toy
def test_quick():
    eng=DarwinacciEngine(toy.init_genome, toy.evaluate, max_cycles=3, pop_size=32, seed=42)
    champ=eng.run(max_cycles=3)
    assert champ is not None and champ.score >= -999
    print("[OK] Darwinacci quick test:", round(champ.score, 4) if champ else "None")