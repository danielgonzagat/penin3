#!/bin/bash
# ⚡ COMANDO ÚNICO FINAL - COPIE E COLE TUDO
# Tempo: 2-3 minutos
# Resultado: Sistema limpo e treinando corretamente

cd /root

echo "════════════════════════════════════════════════════════════════"
echo "⚡ ATIVAÇÃO FINAL - FASE CRÍTICA"
echo "════════════════════════════════════════════════════════════════"

# PASSO 1: Limpar TUDO (começar do zero limpo)
echo "🧹 Limpando processos antigos..."
pkill -9 -f "EMERGENCE_CATALYST" 2>/dev/null || true
pkill -9 -f "brain_daemon_real_env" 2>/dev/null || true
pkill -9 -f "darwin_runner" 2>/dev/null || true
pkill -9 -f "cartpole_dqn" 2>/dev/null || true
sleep 3
echo "   ✅ Limpo"

# PASSO 2: Iniciar Darwin (limpo)
echo "🧬 Iniciando Darwin..."
if ! netstat -tuln 2>/dev/null | grep -q ":9092 "; then
    nohup timeout 72h python3 -u darwin_runner.py > darwin_final.log 2>&1 &
    echo $! > darwin.pid
    echo "   ✅ Darwin: PID $(cat darwin.pid)"
    sleep 3
else
    echo "   ⚠️ Porta 9092 ocupada, pulando Darwin"
fi

# PASSO 3: CartPole DQN (QUICK WIN - 3-4h)
echo "🎯 Iniciando CartPole DQN otimizado..."

cat > /tmp/dqn_quickwin.py << 'DQNCODE'
#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, json
from collections import deque
try:
    import gymnasium as gym
except:
    import gym

class DQN(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,a))
    def forward(self, x): return self.net(x)

env = gym.make('CartPole-v1')
s_dim, a_dim = env.observation_space.shape[0], env.action_space.n
policy, target = DQN(s_dim,a_dim), DQN(s_dim,a_dim)
target.load_state_dict(policy.state_dict())
opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
buf = deque(maxlen=100000)
rewards, steps = [], 0

print("🎯 DQN CartPole - Quick Win Edition")
for ep in range(600):
    state, _ = env.reset()
    ep_r, done = 0, False
    
    while not done:
        eps = 0.01 + 0.99 * np.exp(-steps/12000)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.append((state, action, reward, next_state, float(done)))
        state, ep_r, steps = next_state, ep_r + reward, steps + 1
        
        if len(buf) >= 1000 and steps % 4 == 0:
            batch = random.sample(buf, 64)
            s,a,r,ns,d = zip(*batch)
            s,a,r,ns,d = torch.FloatTensor(s), torch.LongTensor(a), torch.FloatTensor(r), torch.FloatTensor(ns), torch.FloatTensor(d)
            
            curr_q = policy(s).gather(1, a.unsqueeze(1))
            with torch.no_grad():
                next_q = target(ns).max(1)[0]
                targ_q = r + 0.99 * next_q * (1-d)
            
            loss = F.smooth_l1_loss(curr_q.squeeze(), targ_q)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
            opt.step()
        
        if steps % 250 == 0:
            target.load_state_dict(policy.state_dict())
    
    rewards.append(ep_r)
    if (ep+1) % 20 == 0:
        avg = np.mean(rewards[-100:]) if len(rewards)>=100 else np.mean(rewards)
        print(f"Ep {ep+1}: reward={ep_r:.0f}, avg100={avg:.1f}, eps={eps:.3f}")
        if avg >= 195: print(f"🎉 SOLVED!"); break

avg_final = np.mean(rewards[-100:])
print(f"\n{'✅ SOLVED' if avg_final >= 195 else '⚠️ Almost'}: avg={avg_final:.1f}")
torch.save(policy.state_dict(), '/root/dqn_solved.pt')
with open('/root/dqn_result.json','w') as f: json.dump({'avg':float(avg_final),'solved':avg_final>=195},f)
DQNCODE

python3 /tmp/dqn_quickwin.py > /root/dqn_win.log 2>&1 &
echo $! > dqn.pid
echo "   ✅ DQN treinando: PID $(cat dqn.pid)"

# PASSO 4: UnifiedBrain (limpo, sem duplicatas)
echo "🧠 Iniciando UnifiedBrain FINAL..."
cd /root/UNIFIED_BRAIN

# NÃO iniciar se massive_replay já rodando bem
if ps aux | grep -q "run_massive_replay.py" && ! grep -q "reward=0.0" /root/continuous_monitor.log 2>/dev/null; then
    echo "   ℹ️ Massive replay já ativo, não duplicar"
else
    # Iniciar clean
    nohup python3 brain_daemon_real_env.py > /root/brain_CLEAN.log 2>&1 &
    echo $! > /root/brain_clean.pid
    echo "   ✅ Brain: PID $(cat /root/brain_clean.pid)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ SISTEMA ATIVO"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 MONITORAR:"
echo ""
echo "# Dashboard completo (atualiza a cada 60s)"
echo "watch -n 60 'bash -c \"echo CartPole: && tail -3 /root/dqn_win.log | grep Ep && echo && echo Brain: && tail -3 /root/brain_CLEAN.log | grep Ep\"'"
echo ""
echo "# Ou individual:"
echo "tail -f /root/dqn_win.log | grep Ep          # CartPole"
echo "tail -f /root/brain_CLEAN.log | grep Ep      # Brain"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 VERIFICAR EM 4 HORAS:"
echo ""
echo "grep SOLVED /root/dqn_win.log && cat /root/dqn_result.json"
echo ""
echo "════════════════════════════════════════════════════════════════"