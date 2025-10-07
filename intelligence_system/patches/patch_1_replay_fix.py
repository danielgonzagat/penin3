#!/usr/bin/env python3
from pathlib import Path

target = Path("/root/intelligence_system/core/system_v7_ultimate.py")
code = target.read_text()

# REMOVER contador errado (linhas 734-741)
old = """            # PPO update (when batch is ready)
            if len(self.rl_agent.states) >= self.rl_agent.batch_size:
                # Record how many transitions we are about to train on
                _used_transitions = len(self.rl_agent.states)
                loss_info = self.rl_agent.update(next_state if not done else state)
                # Increment replay-trained sample counter
                try:
                    self._replay_trained_count += max(_used_transitions, self.rl_agent.batch_size)
                except Exception:
                    self._replay_trained_count += self.rl_agent.batch_size"""

new = """            # PPO update (when batch is ready)
            if len(self.rl_agent.states) >= self.rl_agent.batch_size:
                loss_info = self.rl_agent.update(next_state if not done else state)"""

if old in code:
    code = code.replace(old, new)
    target.write_text(code)
    print("✅ Patch #1 aplicado com sucesso")
else:
    print("⚠️  Patch #1: bloco não encontrado (já aplicado?)")
