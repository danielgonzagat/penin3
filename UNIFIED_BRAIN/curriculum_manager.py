#!/usr/bin/env python3
"""
🎓 CURRICULUM MANAGER
Progressive difficulty learning
"""

__version__ = "1.0.0"

class CurriculumManager:
    """
    Manage progressive difficulty stages.
    Advances when agent masters current stage.
    
    BLOCO 3 - TAREFA 36: Multi-env curriculum support
    """
    
    def __init__(self, env_name: str = 'CartPole-v1'):
        self.env_name = env_name
        self.stage = 0
        
        # Different curricula for different environments
        self.curricula = {
            'CartPole-v1': [
                {'name': 'beginner', 'threshold': 50, 'description': 'Basic control'},
                {'name': 'intermediate', 'threshold': 100, 'description': 'Consistent'},
                {'name': 'advanced', 'threshold': 150, 'description': 'Near-optimal'},
                {'name': 'expert', 'threshold': 195, 'description': 'Solved'}
            ],
            'MountainCar-v0': [
                {'name': 'beginner', 'threshold': -180, 'description': 'Random exploration'},
                {'name': 'intermediate', 'threshold': -150, 'description': 'Learning momentum'},
                {'name': 'advanced', 'threshold': -120, 'description': 'Efficient climbing'},
                {'name': 'expert', 'threshold': -110, 'description': 'Solved'}
            ],
            'Acrobot-v1': [
                {'name': 'beginner', 'threshold': -400, 'description': 'Basic swinging'},
                {'name': 'intermediate', 'threshold': -200, 'description': 'Coordinated swings'},
                {'name': 'advanced', 'threshold': -100, 'description': 'Near-optimal'},
                {'name': 'expert', 'threshold': -90, 'description': 'Solved'}
            ]
        }
        
        # Get stages for current env (default to CartPole)
        self.stages = self.curricula.get(env_name, self.curricula['CartPole-v1'])
    
    def should_advance(self, avg_reward_100):
        """
        Check if should move to next stage.
        
        Args:
            avg_reward_100: Average reward over last 100 episodes
        
        Returns:
            bool: True if advanced to next stage
        """
        if self.stage >= len(self.stages) - 1:
            return False  # Already at max stage
        
        current = self.stages[self.stage]
        if avg_reward_100 >= current['threshold']:
            self.stage += 1
            return True
        
        return False
    
    def get_current_stage(self):
        """Get current stage info"""
        return self.stages[self.stage]
    
    def get_progress(self, avg_reward_100):
        """Get progress in current stage (0-1)"""
        if self.stage >= len(self.stages) - 1:
            return 1.0  # Max stage
        
        current = self.stages[self.stage]
        next_stage = self.stages[self.stage + 1]
        
        if avg_reward_100 <= current['threshold']:
            return 0.0
        elif avg_reward_100 >= next_stage['threshold']:
            return 1.0
        else:
            range_size = next_stage['threshold'] - current['threshold']
            progress = (avg_reward_100 - current['threshold']) / range_size
            return min(1.0, max(0.0, progress))
    
    def get_stats(self):
        """Get curriculum statistics"""
        return {
            'env_name': self.env_name,
            'current_stage': self.stage,
            'stage_name': self.stages[self.stage]['name'],
            'total_stages': len(self.stages),
            'progress_pct': (self.stage / (len(self.stages) - 1)) * 100 if len(self.stages) > 1 else 100
        }
    
    def switch_env(self, new_env_name: str):
        """
        BLOCO 3 - TAREFA 36: Switch to new environment curriculum
        
        Args:
            new_env_name: New environment name
        """
        if new_env_name != self.env_name:
            self.env_name = new_env_name
            self.stages = self.curricula.get(new_env_name, self.curricula['CartPole-v1'])
            self.stage = 0  # Reset to beginner
            return True
        return False


if __name__ == "__main__":
    # Test
    print("Testing Curriculum Manager...")
    
    cm = CurriculumManager()
    
    # Simulate progression
    rewards = [20, 40, 60, 80, 110, 140, 160, 195]
    
    for r in rewards:
        print(f"\nReward: {r}")
        print(f"  Stage: {cm.get_current_stage()['name']}")
        print(f"  Progress: {cm.get_progress(r):.1%}")
        
        if cm.should_advance(r):
            new_stage = cm.get_current_stage()
            print(f"  ✅ Advanced to: {new_stage['name']}")
    
    print(f"\nFinal stats: {cm.get_stats()}")
    print("✅ Curriculum Manager OK!")
