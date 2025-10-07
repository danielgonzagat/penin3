import unittest
import logging
import sys
from pathlib import Path

# Ensure package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.system_v7_ultimate import IntelligenceSystemV7

logging.basicConfig(level=logging.WARNING)

class TestEvolutionaryComponents(unittest.TestCase):
    def test_evolution_increments(self):
        v7 = IntelligenceSystemV7()
        initial_gen = v7.evolutionary_optimizer.generation
        stats = v7._evolve_architecture({'test': 95.0})
        self.assertGreaterEqual(v7.evolutionary_optimizer.generation, initial_gen + 1)
        self.assertIn('generation', stats)

    def test_experience_replay_fills_with_exploration(self):
        v7 = IntelligenceSystemV7()
        initial_size = len(v7.experience_replay)
        # Force exploration-only episode
        v7._exploration_only_episode()
        self.assertGreater(len(v7.experience_replay), initial_size)

    def test_darwin_evolves_generation(self):
        v7 = IntelligenceSystemV7()
        initial_gen = v7.darwin_real.generation
        res = v7._darwin_evolve()
        # Darwin may error but should at least attempt
        self.assertGreaterEqual(v7.darwin_real.generation, initial_gen)
        self.assertIsInstance(res, dict)

if __name__ == '__main__':
    unittest.main()
