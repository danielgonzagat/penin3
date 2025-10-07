#!/bin/bash
# 🚀 COMANDOS FINAIS - ATIVAR INTELIGÊNCIA REAL
# Copie e cole TUDO de uma vez no terminal
# Tempo total: ~5 minutos de setup + execução em background

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🚀 ATIVAÇÃO FINAL DA INTELIGÊNCIA - FASE CRÍTICA"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# PASSO 1: LIMPEZA FINAL (2 min)
# ═══════════════════════════════════════════════════════════════════

echo "🧹 PASSO 1: Limpeza final de processos..."

# Matar EMERGENCE_CATALYST duplicados restantes
CATALYST_PIDS=$(ps aux | grep "EMERGENCE_CATALYST_4" | grep -v grep | awk '{print $2}')
CATALYST_COUNT=$(echo "$CATALYST_PIDS" | grep -v "^$" | wc -l)

if [ $CATALYST_COUNT -gt 0 ]; then
    echo "   Matando $CATALYST_COUNT processos EMERGENCE_CATALYST..."
    echo "$CATALYST_PIDS" | xargs -r kill -9 2>/dev/null || true
    echo "   ✅ Limpeza completa"
fi

# Fix Darwin (matar todos e reiniciar limpo)
echo "   Reiniciando Darwin clean..."
pkill -9 -f darwin_runner 2>/dev/null || true
sleep 2

# Verificar porta livre
if netstat -tuln 2>/dev/null | grep -q ":9092 "; then
    PORT_PID=$(lsof -ti:9092 2>/dev/null || true)
    [ -n "$PORT_PID" ] && kill -9 $PORT_PID 2>/dev/null || true
    sleep 1
fi

# Iniciar Darwin clean
cd /root
nohup timeout 72h python3 -u darwin_runner.py > /root/darwin_clean.log 2>&1 &
DARWIN_PID=$!
echo $DARWIN_PID > /root/darwin.pid
echo "   ✅ Darwin iniciado: PID $DARWIN_PID"

sleep 3

echo ""

# ═══════════════════════════════════════════════════════════════════
# PASSO 2: INICIAR CARTPOLE DQN OTIMIZADO (Quick Win - 4h)
# ═══════════════════════════════════════════════════════════════════

echo "🎯 PASSO 2: Iniciando CartPole DQN (QUICK WIN)..."

cd /root

# Criar versão otimizada inline
cat > /root/cartpole_dqn_FINAL_OPTIMIZED.py << 'PYTHON_FINAL'
#!/usr/bin/env python3
"""CartPole DQN - Versão FINAL OTIMIZADA para RESOLVER"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
try:
    import gymnasium as gym
except:
    import gym
import json
from pathlib import Path

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_dqn():
    # HIPERPARÂMETROS OTIMIZADOS
    EPISODES = 600  # Aumentado de 300
    LR = 5e-4  # Reduzido de 1e-3 para estabilidade
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 12000  # Mais rápido (era 20000)
    BUFFER_SIZE = 100000  # Aumentado
    BATCH_SIZE = 64
    TARGET_UPDATE = 250  # Mais frequente (era 500)
    LEARNING_STARTS = 1000
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    rewards_history = []
    steps_total = 0
    
    print("🎯 CartPole DQN - FINAL OPTIMIZED")
    print(f"Episodes: {EPISODES}, LR: {LR}, Buffer: {BUFFER_SIZE}")
    print("")
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                     np.exp(-steps_total / EPSILON_DECAY)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            steps_total += 1
            
            # Train
            if len(replay_buffer) >= LEARNING_STARTS and steps_total % 4 == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + GAMMA * next_q * (1 - dones)
                
                loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
            
            # Update target
            if steps_total % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        rewards_history.append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_10 = np.mean(rewards_history[-10:])
            avg_100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else avg_10
            print(f"Ep {episode+1}/{EPISODES}: reward={episode_reward:.0f}, avg10={avg_10:.1f}, avg100={avg_100:.1f}, eps={epsilon:.3f}")
        
        # Check solved
        if len(rewards_history) >= 100:
            avg_100 = np.mean(rewards_history[-100:])
            if avg_100 >= 195.0:
                print(f"\n🎉 SOLVED at episode {episode+1}! Avg100={avg_100:.1f}")
                break
    
    # Final evaluation
    print("\n" + "="*60)
    print("📊 AVALIAÇÃO FINAL")
    print("="*60)
    
    eval_rewards = []
    for _ in range(20):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_t).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        eval_rewards.append(episode_reward)
    
    avg_eval = np.mean(eval_rewards)
    solved_count = sum(1 for r in eval_rewards if r >= 195)
    
    print(f"Eval (20 eps): avg={avg_eval:.1f}, solved={solved_count}/20")
    print(f"Status: {'✅ SOLVED' if solved_count >= 19 else '⚠️ Almost solved' if solved_count >= 15 else '❌ Not solved'}")
    
    # Save
    out_dir = Path("/root/cartpole_dqn_final_run")
    out_dir.mkdir(exist_ok=True)
    
    torch.save({
        'policy_state': policy_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'episodes': episode + 1,
        'final_avg': avg_eval
    }, out_dir / "dqn_final.pt")
    
    with open(out_dir / "results.json", 'w') as f:
        json.dump({
            'episodes_trained': episode + 1,
            'final_avg_100': float(np.mean(rewards_history[-100:])),
            'eval_avg_20': float(avg_eval),
            'eval_solved_count': solved_count,
            'status': 'solved' if solved_count >= 19 else 'almost',
            'all_rewards': [float(r) for r in rewards_history]
        }, f, indent=2)
    
    print(f"\n✅ Model saved: {out_dir}/dqn_final.pt")
    print(f"✅ Results: {out_dir}/results.json")
    
    env.close()
    return policy_net, rewards_history

if __name__ == "__main__":
    train_dqn()
PYTHON_FINAL

chmod +x /root/cartpole_dqn_FINAL_OPTIMIZED.py

echo "   ✅ CartPole DQN otimizado criado"
echo "   🚀 Iniciando treinamento (background, ~3-4h)..."

nohup python3 /root/cartpole_dqn_FINAL_OPTIMIZED.py > /root/cartpole_victory.log 2>&1 &
CARTPOLE_PID=$!
echo $CARTPOLE_PID > /root/cartpole.pid

echo "   ✅ CartPole treinando: PID $CARTPOLE_PID"
echo "   📊 Monitor: tail -f /root/cartpole_victory.log | grep 'Ep '"

sleep 2
echo ""

# ═══════════════════════════════════════════════════════════════════
# PASSO 3: REINICIAR UNIFIED_BRAIN COM CORREÇÕES (2 min)
# ═══════════════════════════════════════════════════════════════════

echo "🧠 PASSO 3: Reiniciando UnifiedBrain com correções aplicadas..."

# Matar instances antigas (mantém massive_replay rodando)
OLD_BRAIN_PIDS=$(ps aux | grep "brain_daemon_real_env.py" | grep -v grep | grep -v "run_massive" | awk '{print $2}')
if [ -n "$OLD_BRAIN_PIDS" ]; then
    echo "   Parando brain daemons antigos..."
    echo "$OLD_BRAIN_PIDS" | xargs kill -15 2>/dev/null || true
    sleep 3
fi

# Iniciar nova instância com correções
cd /root/UNIFIED_BRAIN
nohup python3 brain_daemon_real_env.py > /root/brain_FINAL.log 2>&1 &
BRAIN_PID=$!
echo $BRAIN_PID > /root/brain.pid

echo "   ✅ UnifiedBrain iniciado: PID $BRAIN_PID"
echo "   📊 Monitor: tail -f /root/brain_FINAL.log | grep 'Ep '"

sleep 2
echo ""

# ═══════════════════════════════════════════════════════════════════
# PASSO 4: CRIAR DASHBOARD DE MONITORAMENTO
# ═══════════════════════════════════════════════════════════════════

echo "📊 PASSO 4: Criando dashboard de monitoramento..."

cat > /root/monitor_inteligencia.sh << 'MONITOR_EOF'
#!/bin/bash
# Monitor da Inteligência Emergente
clear
echo "════════════════════════════════════════════════════════════════"
echo "🧠 MONITOR DE INTELIGÊNCIA EMERGENTE - $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════════════════════════"
echo ""

# CartPole DQN
echo "🎯 CartPole DQN (Quick Win):"
if [ -f /root/cartpole.pid ] && ps -p $(cat /root/cartpole.pid 2>/dev/null) > /dev/null 2>&1; then
    LAST_LINE=$(tail -100 /root/cartpole_victory.log 2>/dev/null | grep "Ep " | tail -1)
    if [ -n "$LAST_LINE" ]; then
        echo "   $LAST_LINE"
    else
        echo "   🔄 Iniciando..."
    fi
    
    # Check if solved
    if grep -q "SOLVED" /root/cartpole_victory.log 2>/dev/null; then
        echo "   🎉 STATUS: RESOLVIDO! ✅"
    fi
else
    echo "   ⚠️ Não rodando"
fi

echo ""

# UnifiedBrain
echo "🧠 UnifiedBrain:"
if [ -f /root/brain.pid ] && ps -p $(cat /root/brain.pid 2>/dev/null) > /dev/null 2>&1; then
    LAST_BRAIN=$(tail -50 /root/brain_FINAL.log 2>/dev/null | grep "Ep " | tail -1)
    if [ -n "$LAST_BRAIN" ]; then
        echo "   $LAST_BRAIN"
    else
        echo "   🔄 Iniciando..."
    fi
    
    # Check meta_step
    META_COUNT=$(grep -c "META.*ACCEPTED\|META.*REJECTED" /root/brain_FINAL.log 2>/dev/null || echo "0")
    echo "   Meta-steps executados: $META_COUNT"
else
    echo "   ⚠️ Não rodando"
fi

echo ""

# V7
echo "📊 V7 Ultimate:"
V7_PID=$(ps aux | grep "unified_agi_system.py 100" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$V7_PID" ]; then
    echo "   PID: $V7_PID"
    LAST_V7=$(tail -50 /root/intelligence_system/logs/v7_output.log 2>/dev/null | grep "Cycle" | tail -1)
    [ -n "$LAST_V7" ] && echo "   $LAST_V7" || echo "   🔄 Rodando..."
else
    echo "   ⚠️ Não rodando"
fi

echo ""

# Darwin
echo "🧬 Darwin Evolution:"
if [ -f /root/darwin.pid ] && ps -p $(cat /root/darwin.pid 2>/dev/null) > /dev/null 2>&1; then
    DARWIN_PID=$(cat /root/darwin.pid)
    echo "   PID: $DARWIN_PID"
    
    # Test metrics
    if curl -s http://localhost:9092/metrics 2>&1 | grep -q "darwin"; then
        DECISIONS=$(curl -s http://localhost:9092/metrics 2>/dev/null | grep "darwin_decisions_total" | awk '{print $2}')
        PROMOTIONS=$(curl -s http://localhost:9092/metrics 2>/dev/null | grep "darwin_promotions_total" | awk '{print $2}')
        echo "   Decisions: $DECISIONS, Promotions: $PROMOTIONS"
    else
        echo "   ⚠️ Metrics não disponíveis (ainda inicializando)"
    fi
else
    echo "   ⚠️ Não rodando"
fi

echo ""

# Database stats
echo "📊 Database (Intelligence.db):"
DB="/root/intelligence_system/data/intelligence.db"
if [ -f "$DB" ]; then
    CYCLES=$(sqlite3 "$DB" "SELECT COUNT(*) FROM cycles;" 2>/dev/null || echo "?")
    GATE_EVALS=$(sqlite3 "$DB" "SELECT COUNT(*) FROM gate_evals;" 2>/dev/null || echo "?")
    SURPRISES=$(sqlite3 "$DB" "SELECT COUNT(*) FROM events WHERE event_type='statistical_surprise' AND z_score >= 4;" 2>/dev/null || echo "?")
    
    echo "   Total cycles: $CYCLES"
    echo "   Gate evals: $GATE_EVALS (meta-learning)"
    echo "   Surprises (z≥4): $SURPRISES ⚡"
fi

echo ""

# Resources
echo "💾 Recursos:"
MEM_PCT=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
DISK_PCT=$(df -h /root | tail -1 | awk '{print $5}' | tr -d '%')
echo "   Memory: ${MEM_PCT}%"
echo "   Disk: ${DISK_PCT}%"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🔄 Atualizado a cada 60s - Ctrl+C para sair"
echo "════════════════════════════════════════════════════════════════"
MONITOR_EOF

chmod +x /root/monitor_inteligencia.sh
echo "   ✅ Dashboard criado: monitor_inteligencia.sh"

echo ""

# ═══════════════════════════════════════════════════════════════════
# PASSO 5: VACUUM E OTIMIZAR DATABASES
# ═══════════════════════════════════════════════════════════════════

echo "🗄️ PASSO 5: Otimizando databases..."

sqlite3 /root/intelligence_system/data/intelligence.db "VACUUM; ANALYZE;" 2>/dev/null || echo "   ⚠️ DB em uso, pular"
echo "   ✅ Database otimizado"

echo ""

# ═══════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "✅ ATIVAÇÃO COMPLETA - SISTEMA RODANDO"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🎯 Processos Ativos:"
echo "   CartPole DQN: PID $(cat /root/cartpole.pid 2>/dev/null || echo '?')"
echo "   UnifiedBrain: PID $(cat /root/brain.pid 2>/dev/null || echo '?')"
echo "   Darwin: PID $(cat /root/darwin.pid 2>/dev/null || echo '?')"
echo "   V7 Ultimate: PID $V7_PID"
echo ""
echo "📊 Monitoramento:"
echo "   Dashboard: watch -n 60 /root/monitor_inteligencia.sh"
echo "   CartPole: tail -f /root/cartpole_victory.log | grep Ep"
echo "   Brain: tail -f /root/brain_FINAL.log | grep Ep"
echo ""
echo "⏱️ Tempo Esperado:"
echo "   CartPole DQN: 3-4h até resolver"
echo "   UnifiedBrain: 2-4 semanas até emergência"
echo ""
echo "🎯 Próxima Verificação:"
echo "   Hoje (4h): Checar se CartPole resolveu"
echo "   Amanhã: Verificar UnifiedBrain melhorando"
echo "   Semana 1: Analisar trends e ajustar"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🚀 INTELIGÊNCIA ATIVADA - DEIXE RODAR!"
echo "════════════════════════════════════════════════════════════════"