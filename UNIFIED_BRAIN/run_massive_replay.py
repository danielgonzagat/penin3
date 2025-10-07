#!/usr/bin/env python3
"""
🚀 MASSIVE REPLAY RUNNER - 1000+ GERAÇÕES
Executa replay massivo com checkpoints regulares e WORM por geração
"""
import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import time
import json
from pathlib import Path
from brain_daemon_real_env import RealEnvironmentBrainV3
from brain_logger import brain_logger

def run_massive_replay(
    total_generations=1000,
    episodes_per_gen=10,
    checkpoint_every=50,
    env_name='CartPole-v1',
    learning_rate=3e-4,
):
    """
    Executa replay massivo com:
    - Checkpoints regulares a cada N gerações
    - WORM logging por geração
    - Meta-step a cada 10 gerações
    - Curriculum advancement automático
    """
    brain_logger.info("=" * 80)
    brain_logger.info(f"🚀 MASSIVE REPLAY INICIANDO")
    brain_logger.info(f"   Total gerações: {total_generations}")
    brain_logger.info(f"   Episodes/geração: {episodes_per_gen}")
    brain_logger.info(f"   Checkpoint a cada: {checkpoint_every} gens")
    brain_logger.info(f"   Environment: {env_name}")
    brain_logger.info("=" * 80)
    
    # Setup output directory
    output_dir = Path('/root/massive_replay_output')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar daemon
    brain_logger.info("🔧 Inicializando daemon...")
    daemon = RealEnvironmentBrainV3(
        env_name=env_name,
        learning_rate=learning_rate,
        use_gpu=False
    )
    daemon.initialize()
    
    # WORM file
    worm_file = output_dir / 'massive_replay_worm.jsonl'
    summary_file = output_dir / 'summary.json'
    
    # Stats tracking
    all_rewards = []
    meta_acceptances = []
    start_time = time.time()
    
    for gen in range(total_generations):
        gen_start = time.time()
        gen_rewards = []
        
        # Rodar N episodes por geração
        for ep in range(episodes_per_gen):
            try:
                reward = daemon.run_episode()
                gen_rewards.append(float(reward))
                all_rewards.append(float(reward))
            except KeyboardInterrupt:
                brain_logger.warning("⚠️ Interrompido pelo usuário")
                raise
            except Exception as e:
                brain_logger.error(f"❌ Episode erro: {e}")
                gen_rewards.append(0.0)
        
        gen_time = time.time() - gen_start
        avg_reward = sum(gen_rewards) / max(1, len(gen_rewards))
        best_reward = max(gen_rewards) if gen_rewards else 0.0
        
        # Log geração
        brain_logger.info(
            f"📊 GEN {gen+1:04d}/{total_generations}: "
            f"avg={avg_reward:.2f}, "
            f"best={best_reward:.2f}, "
            f"time={gen_time:.1f}s, "
            f"total_eps={len(all_rewards)}"
        )
        
        # WORM entry por geração
        worm_entry = {
            'generation': gen + 1,
            'episodes': len(gen_rewards),
            'avg_reward': avg_reward,
            'best_reward': best_reward,
            'worst_reward': min(gen_rewards) if gen_rewards else 0.0,
            'gen_time_s': gen_time,
            'timestamp': time.time(),
            'total_episodes': len(all_rewards),
        }
        
        with open(worm_file, 'a') as f:
            f.write(json.dumps(worm_entry) + '\n')
        
        # Checkpoint regular
        if (gen + 1) % checkpoint_every == 0:
            ckpt_path = output_dir / f'checkpoint_gen_{gen+1:04d}.pt'
            try:
                daemon.save_checkpoint(str(ckpt_path))
                brain_logger.info(f"💾 Checkpoint: {ckpt_path.name}")
            except Exception as e:
                brain_logger.error(f"❌ Checkpoint erro: {e}")
        
        # Meta-step a cada 10 gerações
        if (gen + 1) % 10 == 0:
            if hasattr(daemon, 'controller') and daemon.controller:
                try:
                    brain_logger.info(f"🧠 [META] Executando meta_step (gen {gen+1})")
                    accepted = daemon.controller.meta_step()
                    meta_acceptances.append(accepted)
                    
                    result_str = '✅ ACEITO' if accepted else '❌ REJEITADO'
                    brain_logger.info(f"🧠 [META] {result_str}")
                    
                    # Log acceptance rate
                    if meta_acceptances:
                        accept_rate = sum(meta_acceptances) / len(meta_acceptances)
                        brain_logger.info(f"📈 [META] Taxa de aceitação: {accept_rate*100:.1f}%")
                
                except Exception as e:
                    brain_logger.error(f"❌ [META] Erro: {e}")
        
        # Progress report a cada 100 gerações
        if (gen + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_all = sum(all_rewards) / max(1, len(all_rewards))
            recent_100 = all_rewards[-1000:] if len(all_rewards) >= 1000 else all_rewards
            avg_recent = sum(recent_100) / max(1, len(recent_100))
            
            brain_logger.info("=" * 80)
            brain_logger.info(f"📊 PROGRESS REPORT - Geração {gen+1}")
            brain_logger.info(f"   Tempo decorrido: {elapsed/3600:.1f}h")
            brain_logger.info(f"   Episodes totais: {len(all_rewards)}")
            brain_logger.info(f"   Avg reward (geral): {avg_all:.2f}")
            brain_logger.info(f"   Avg reward (recente): {avg_recent:.2f}")
            brain_logger.info(f"   Melhor reward: {max(all_rewards):.2f}")
            brain_logger.info(f"   Meta acceptances: {sum(meta_acceptances)}/{len(meta_acceptances)}")
            brain_logger.info("=" * 80)
    
    # Final summary
    total_time = time.time() - start_time
    final_avg = sum(all_rewards) / max(1, len(all_rewards))
    
    summary = {
        'total_generations': total_generations,
        'total_episodes': len(all_rewards),
        'total_time_hours': total_time / 3600,
        'final_avg_reward': final_avg,
        'best_reward': max(all_rewards) if all_rewards else 0.0,
        'meta_acceptance_rate': sum(meta_acceptances) / max(1, len(meta_acceptances)),
        'rewards_all': all_rewards,
        'timestamp': time.time(),
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final checkpoint
    final_ckpt = output_dir / 'final_checkpoint.pt'
    daemon.save_checkpoint(str(final_ckpt))
    
    brain_logger.info("=" * 80)
    brain_logger.info("✅ MASSIVE REPLAY COMPLETO")
    brain_logger.info(f"   Gerações: {total_generations}")
    brain_logger.info(f"   Episodes: {len(all_rewards)}")
    brain_logger.info(f"   Tempo: {total_time/3600:.2f}h")
    brain_logger.info(f"   Avg reward final: {final_avg:.2f}")
    brain_logger.info(f"   Output: {output_dir}")
    brain_logger.info("=" * 80)
    
    return summary

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Massive Replay Runner')
    parser.add_argument('--generations', type=int, default=1000, help='Total generations')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per generation')
    parser.add_argument('--checkpoint-every', type=int, default=50, help='Checkpoint frequency')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        summary = run_massive_replay(
            total_generations=args.generations,
            episodes_per_gen=args.episodes,
            checkpoint_every=args.checkpoint_every,
            env_name=args.env,
            learning_rate=args.lr,
        )
        
        print("\n✅ SUCESSO!")
        print(f"📊 Avg reward final: {summary['final_avg_reward']:.2f}")
        print(f"⏱️  Tempo total: {summary['total_time_hours']:.2f}h")
        print(f"📁 Resultados em: /root/massive_replay_output/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
        print("💾 Progresso salvo em: /root/massive_replay_output/")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()