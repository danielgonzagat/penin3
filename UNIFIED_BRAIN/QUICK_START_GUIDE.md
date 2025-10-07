# 🚀 UNIFIED BRAIN - GUIA DE USO RÁPIDO

---

## ⚡ START EM 3 COMANDOS

```bash
cd /root/UNIFIED_BRAIN

# 1. Conectar neurônios (se ainda não fez)
python3 connect_all_neurons.py

# 2. Testar o cérebro
python3 -c "
from unified_brain_core import UnifiedBrain
import torch

brain = UnifiedBrain(H=1024)
brain.load_snapshot('snapshots/brain_snapshot_1759541433.pt')

z = torch.randn(1, 1024)
z_out = brain(z, num_steps=4)
print('Cérebro funcionando!')
print(brain.get_metrics_summary())
"

# 3. Integrar com seu sistema (exemplo)
python3 test_integration.py
```

---

## 📦 O QUE VOCÊ TEM AGORA

### ✅ 754 Neurônios Conectados
- **254 Darwin Gen45** (testados, CORE)
- **500 API neurons** (experimentais, SOUP)

### ✅ Sistema Funcional
- Espaço latente Z (1024 dims)
- Router IA³ adaptativo
- Adapters treinados
- Snapshot salvo

### ✅ Módulos Prontos
- `unified_brain_core.py` - Cérebro
- `brain_router.py` - Roteador
- `brain_system_integration.py` - V7/PENIN-Ω/Darwin

---

## 🎯 CASOS DE USO

### 1️⃣ **Usar como Encoder**

```python
from unified_brain_core import UnifiedBrain
import torch

brain = UnifiedBrain(H=1024)
brain.load_snapshot('snapshots/brain_snapshot_1759541433.pt')

# Suas observações
obs = torch.randn(1, 784)  # Ex: MNIST flattened

# Criar encoder simples
encoder = torch.nn.Linear(784, 1024)
z_in = encoder(obs)

# Processar no cérebro
z_out = brain(z_in, num_steps=4)

# z_out agora é representação rica processada por 254 neurônios!
```

### 2️⃣ **Integrar com Reinforcement Learning**

```python
from brain_system_integration import BrainV7Bridge
import torch

# Cria bridge
bridge = BrainV7Bridge(brain, obs_dim=4, act_dim=2)

# Loop de treinamento
for episode in range(1000):
    obs = env.reset()
    done = False
    
    while not done:
        # Brain decide ação
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)
        action_logits, value, z = bridge(obs_tensor)
        
        # Amostra ação
        action = torch.multinomial(torch.softmax(action_logits, dim=1), 1).item()
        
        # Executa
        obs, reward, done, _ = env.step(action)
        
        # Atualiza competence dos neurônios (opcional)
        brain.router.update_competence(0, reward)
```

### 3️⃣ **Modular com PENIN-Ω**

```python
from brain_system_integration import BrainPENINOmegaInterface

# Interface
penin = BrainPENINOmegaInterface(brain)

# A cada passo, atualiza sinais
penin_metrics = {
    'L_infinity': 0.6,
    'CAOS_plus': 0.8,
    'SR_Omega_infinity': 1.2
}
penin.update_signals(penin_metrics)

# PENIN-Ω agora modula:
# - Temperature do router (exploration)
# - Alpha (persistência)
# - Lateral inhibition

# Obter IA³ real
ia3_signal = penin.get_ia3_signal()
print(f"IA³ atual: {ia3_signal:.3f}")
```

### 4️⃣ **Evoluir com Darwin**

```python
from brain_system_integration import BrainDarwinEvolver

darwin = BrainDarwinEvolver(brain)

# Loop de evolução
for generation in range(100):
    # Avalia fitness
    reward = run_episode(brain)  # Sua função
    fitness = darwin.evaluate_fitness(reward)
    
    # Evolui
    darwin.evolve_step(fitness['total'])
    
    print(f"Gen {generation}: fitness={fitness['total']:.3f}")
```

### 5️⃣ **Sistema Completo (V7 + PENIN-Ω + Darwin)**

```python
from brain_system_integration import UnifiedSystemController

# Controller orquestra tudo
controller = UnifiedSystemController(brain)
controller.connect_v7(obs_dim=4, act_dim=2)

# Loop único
for step in range(10000):
    # Observação
    obs = env.reset()
    obs_tensor = torch.tensor(obs).float().unsqueeze(0)
    
    # Métricas PENIN-Ω (calcule do seu sistema)
    penin_metrics = {
        'L_infinity': calculate_L_infinity(),
        'CAOS_plus': calculate_CAOS(),
        'SR_Omega_infinity': calculate_SR()
    }
    
    # Step integrado
    result = controller.step(
        obs=obs_tensor,
        penin_metrics=penin_metrics,
        reward=last_reward
    )
    
    # Usa resultado
    action = sample_action(result['action_logits'])
    ia3 = result['ia3_signal']
    fitness = result['fitness']
```

---

## 🔧 CUSTOMIZAÇÃO

### Adicionar Mais Neurônios

```python
from brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
import torch.nn as nn

# Seu neurônio custom
class MyNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

model = MyNeuron()

# Metadata
meta = NeuronMeta(
    id="my_custom_neuron",
    in_shape=(128,),
    out_shape=(64,),
    dtype=torch.float32,
    device='cpu',
    status=NeuronStatus.ACTIVE,
    source='custom',
    params_count=sum(p.numel() for p in model.parameters()),
    checksum="abc123"
)

# Registra
registered = RegisteredNeuron(meta, model.forward, H=1024)
brain.register_neuron(registered)
brain.initialize_router()  # Reconstrói router
```

### Ajustar Parâmetros

```python
# Mais exploração
brain.router.temperature = 2.0
brain.router.top_k = 128

# Mais persistência
brain.alpha = 0.95

# Menos lateral inhibition
brain.lateral_inhibition = 0.05

# Mais passos de processamento
brain.num_steps = 8
```

### Carregar Mais Neurônios

```python
# Modificar connect_all_neurons.py:
# Linha ~180: Aumentar limite
for filepath in safe_files[:5000]:  # De 500 para 5000
    ...
```

---

## 📊 MONITORAMENTO

### Métricas em Tempo Real

```python
# Durante execução
z_out, info = brain.step(z_in)

print(f"Neurônios ativos: {info['selected_neurons']}")
print(f"Coherence: {info['coherence']:.3f}")
print(f"Novelty: {info['novelty']:.3f}")
print(f"Latency: {info['latency_ms']:.1f}ms")

# Top 5 neurônios contribuindo
for act in info['activations'][:5]:
    print(f"  {act['neuron_id']}: weight={act['weight']:.2f}, contrib={act['contribution']:.2f}")
```

### Estatísticas do Router

```python
stats = brain.router.get_activation_stats()

print(f"Média de ativações: {stats['mean_activations']:.1f}")
print(f"Neurônios ativos: {stats['active_neurons']}")
print(f"Top 10 mais usados: {stats['top_10_indices']}")
```

### 📋 Auditoria rápida - queries canônicas

1) Gradiente (evidência de learning)
```bash
rg "router_competence_grad_norm=" UNIFIED_BRAIN/logs -n || true
curl -s localhost:9109/metrics | rg ubrain_router_comp_grad_norm || true
```

2) Surprises estatísticas (>=5σ)
```bash
rg "HIGH-SIGMA SURPRISE" -n || true
curl -s localhost:9091/metrics | rg ubrain_surprise || true
```

3) Uplift/Retention (Gym suite)
```bash
rg "promotion_gate|gate_eval" UNIFIED_BRAIN/worm.log -n || true
```

4) Competence→Activation link
```bash
rg competence_activation_link UNIFIED_BRAIN/worm.log -n || true
```

### Sumário Completo

```python
summary = brain.get_metrics_summary()

print(f"Total steps: {summary['total_steps']}")
print(f"Total activations: {summary['total_activations']}")
print(f"Avg coherence: {summary['avg_coherence']:.3f}")
print(f"Avg novelty: {summary['avg_novelty']:.3f}")
print(f"Avg latency: {summary['avg_latency_ms']:.1f}ms")
print(f"Neuron counts: {summary['neuron_counts']}")
```

---

## 🛑 TROUBLESHOOTING

### Cérebro travou?

```bash
# Kill-switch
touch /root/STOP_BRAIN

# Verifica status
python3 -c "
from unified_brain_core import UnifiedBrain
brain = UnifiedBrain()
brain.load_snapshot('snapshots/brain_snapshot_1759541433.pt')
print(f'Active: {brain.is_active}')
"

# Reativa
rm /root/STOP_BRAIN
brain.is_active = True
```

### Neurônios não ativam?

```python
# Verifica quantos são ACTIVE
counts = brain.registry.count()
print(f"Active: {counts['active']}")
print(f"Testing: {counts['testing']}")

# Promove TESTING → ACTIVE
for neuron in brain.registry.get_by_status(NeuronStatus.TESTING):
    brain.registry.promote(neuron.meta.id, NeuronStatus.ACTIVE)

# Reinicializa router
brain.initialize_router()
```

### Latência alta?

```python
# Reduz top_k
brain.router.top_k = 32  # De 64 para 32

# Reduz num_steps
brain.num_steps = 2  # De 4 para 2

# Usa CPU (se estava em GPU)
brain.device = 'cpu'
```

### Out of memory?

```python
# Reduz max_neurons
brain.max_neurons = 1000  # De 10000

# Limpa neurônios inativos
inactive = [n for n in brain.registry.neurons.values() 
            if n.meta.activation_count == 0]
print(f"Removendo {len(inactive)} neurônios nunca usados")
```

---

## 💾 SNAPSHOTS

### Salvar Estado

```python
brain.save_snapshot('snapshots/my_checkpoint.pt')
```

### Carregar Estado

```python
brain.load_snapshot('snapshots/my_checkpoint.pt')
```

### Rollback

```python
# Lista snapshots
import os
snapshots = sorted(os.listdir('snapshots'))
print(snapshots)

# Carrega anterior
brain.load_snapshot(f'snapshots/{snapshots[-2]}')
```

---

## 🎯 PRÓXIMOS PASSOS RECOMENDADOS

### Fase 1: Teste Básico ✅
- [x] Conectar neurônios Darwin
- [x] Testar forward
- [x] Salvar snapshot

### Fase 2: Integração (AGORA)
- [ ] Conectar com CartPole
- [ ] Treinar com PPO
- [ ] Medir IA³ real

### Fase 3: Escala
- [ ] Carregar IA3 Gen45 (24k neurônios)
- [ ] Construir topologia small-world
- [ ] Arena de competição

### Fase 4: Emergência
- [ ] Detecção de padrões não-programados
- [ ] Reprodução sexual Darwin × API
- [ ] Descobrir inteligência real

---

## 📚 REFERÊNCIAS

### Arquivos Principais:
- `/root/UNIFIED_BRAIN/unified_brain_core.py`
- `/root/UNIFIED_BRAIN/brain_router.py`
- `/root/UNIFIED_BRAIN/brain_system_integration.py`

### Relatórios:
- `/root/UNIFIED_BRAIN_FINAL_REPORT.md`
- `/root/FINAL_COMPLETE_NEURON_CENSUS.md`
- `/root/CONVERSATION_SUMMARY.md`

### Dados:
- `/root/UNIFIED_BRAIN/inventory.json`
- `/root/UNIFIED_BRAIN/connection_summary.json`
- `/root/UNIFIED_BRAIN/neuron_registry.json`

---

## 🔥 EXPERIMENTO RÁPIDO

```python
#!/usr/bin/env python3
"""Teste rápido do cérebro"""
import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')

from unified_brain_core import UnifiedBrain
import torch

# Carrega
brain = UnifiedBrain(H=1024)
brain.load_snapshot('snapshots/brain_snapshot_1759541433.pt')

# Teste com 10 inputs diferentes
for i in range(10):
    z_in = torch.randn(1, 1024)
    z_out, info = brain.step(z_in, reward=0.5 + i*0.05)
    
    print(f"Step {i+1}: {info['selected_neurons']} neurons, "
          f"coherence={info['coherence']:.3f}, "
          f"novelty={info['novelty']:.3f}")

# Sumário
print("\n" + "="*50)
print(brain.get_metrics_summary())
```

**Execute:** `python3 test_brain.py`

---

🎉 **CÉREBRO PRONTO PARA USO!**

**Dúvidas?** Consulte `/root/UNIFIED_BRAIN_FINAL_REPORT.md`
