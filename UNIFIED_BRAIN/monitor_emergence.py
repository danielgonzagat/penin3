#!/usr/bin/env python3
"""
📊 MONITOR DE EMERGÊNCIA
Monitora aprendizado real em tempo real
"""

import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*80)
print("📊 MONITOR DE EMERGÊNCIA - Real Intelligence")
print("="*80)
print()

# Monitora checkpoint
checkpoint_path = Path("/root/UNIFIED_BRAIN/real_env_checkpoint.json")

print("Monitoring real learning progress...")
print("(Waiting for episodes to complete...)")
print()

last_episode = 0

for iteration in range(60):  # Monitora por 1 minuto
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        stats = data.get('stats', {})
        episode = data.get('episode', 0)
        
        if episode > last_episode:
            last_episode = episode
            
            # Exibe progresso
            rewards = stats.get('rewards', [])
            if rewards:
                recent = rewards[-10:] if len(rewards) >= 10 else rewards
                avg_recent = sum(recent) / len(recent)
                best = stats.get('best_reward', 0)
                progress = stats.get('learning_progress', 0) * 100
                
                print(f"Episode {episode:4d}: "
                      f"reward={rewards[-1]:6.1f}, "
                      f"avg_10={avg_recent:6.1f}, "
                      f"best={best:6.1f}, "
                      f"progress={progress:4.1f}%")
                
                # Detecta sinais de emergência
                if avg_recent > 50:
                    print("   🎊 EMERGÊNCIA: Aprendizado detectado!")
                if avg_recent > 100:
                    print("   🔥 FORTE EMERGÊNCIA: Estratégias complexas!")
                if avg_recent > 195:
                    print("   ⭐ PROBLEMA RESOLVIDO: Inteligência real confirmada!")
    
    time.sleep(1)

print()
print("="*80)

# Cria gráfico se tiver dados
if checkpoint_path.exists():
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    rewards = data.get('stats', {}).get('rewards', [])
    
    if len(rewards) > 5:
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        
        # Moving average
        window = 10
        if len(rewards) >= window:
            ma = [sum(rewards[max(0,i-window):i+1]) / min(i+1, window) for i in range(len(rewards))]
            plt.plot(ma, linewidth=2, label=f'Moving Avg ({window})')
        
        plt.axhline(y=195, color='r', linestyle='--', label='Solved (195)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Real Intelligence Learning Progress')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_path = '/root/UNIFIED_BRAIN/learning_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Learning curve saved: {plot_path}")

print("Monitor complete!")
print("="*80)
