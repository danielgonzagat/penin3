import unittest
import torch
from intelligence_system.extracted_algorithms.maml_engine import MAMLEngine, Task
from intelligence_system.extracted_algorithms.novelty_system import NoveltySystem

class TestMAML(unittest.TestCase):
    def test_maml_backward_sync(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 3))
        engine = MAMLEngine(model, inner_steps=1, first_order=True)
        task = Task(
            support_x=torch.randn(4, 10), support_y=torch.randint(0, 3, (4,)),
            query_x=torch.randn(2, 10), query_y=torch.randint(0, 3, (2,))
        )
        adapted, loss = engine.inner_loop(task, create_graph=False)
        self.assertIsInstance(loss, float)
        hist = engine.meta_train(lambda: [task, task], n_iterations=1, tasks_per_iteration=2)
        self.assertTrue(isinstance(hist, list))

class TestNovelty(unittest.TestCase):
    def test_novelty_reward(self):
        ns = NoveltySystem(k_nearest=5, archive_size=10, novelty_threshold=0.1)
        behavior = torch.tensor([0.5, 0.5]).numpy()
        reward = ns.reward_novelty(behavior, 0.5, 0.3)
        self.assertIsInstance(reward, float)

if __name__ == '__main__':
    unittest.main()
