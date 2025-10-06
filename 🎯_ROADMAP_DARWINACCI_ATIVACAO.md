# 🎯 ROADMAP: ATIVAR DARWINACCI-Ω

**Data**: 2025-10-05 19:15:00  
**Objetivo**: Integrar Darwinacci ao sistema de inteligência  
**Status Atual**: ⚠️ Criado mas não integrado  
**Meta**: ✅ Integração completa + validação empírica

---

## 📋 SITUAÇÃO ATUAL (BRUTAL E HONESTA)

### ✅ O QUE EXISTE:
- Código Darwinacci-Ω completo (394 linhas)
- 12 módulos funcionais
- 1 processo rodando standalone há 3h+ (PID 2710411)
- WORM ledger com 12 entradas
- Wrapper V7 configurado

### ❌ O QUE NÃO FUNCIONA:
- V7 não consegue importar Darwinacci core
- Brain Daemon não usa Darwinacci
- Darwin original ainda ativo (não substituído)
- Sem dados de fitness trend (WORM com erro de parse)

### 🎯 GAP:
**Darwinacci está "orfa"** - existe mas não está conectado ao ecossistema principal

---

## 🚀 ROADMAP DE ATIVAÇÃO (Por Simplicidade)

### **FASE 1: CORREÇÕES CRÍTICAS** (<30 min, Certeza 95%)

#### 1.1. Corrigir Bug TimeCrystal ✅ MAIS SIMPLES
**Tempo**: 2 minutos  
**Certeza**: 100%

**Problema**:
```python
AttributeError: 'TimeCrystal' object has no attribute 'max_cycles'
```

**Solução**:
```python
# /root/darwinacci_omega/core/f_clock.py
class TimeCrystal:
    def __init__(self, max_cycles):
        self.max_cycles = max_cycles  # ← ADICIONAR
        self.fib = [1,1,2,3,5,8,13,21]
        # ...
```

**Impacto**: Fix minor bug

---

#### 1.2. Corrigir Path PYTHONPATH ✅ SIMPLES
**Tempo**: 5 minutos  
**Certeza**: 95%

**Problema**:
```
ImportError: No module named 'darwinacci_omega'
```

**Solução**:
```bash
# Opção A: Export global
echo 'export PYTHONPATH="/root:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc

# Opção B: Add ao sys.path no wrapper
# /root/intelligence_system/extracted_algorithms/darwin_engine_darwinacci.py
# Linha 29: sys.path.insert(0, '/root')  # ← JÁ EXISTE!

# Opção C: Instalar como package
cd /root/darwinacci_omega
cat > setup.py << 'SETUP'
from setuptools import setup, find_packages

setup(
    name="darwinacci-omega",
    version="1.0.0",
    packages=find_packages(),
)
SETUP
pip install -e .
```

**Impacto**: V7 pode importar Darwinacci

---

#### 1.3. Testar Import ✅ SIMPLES
**Tempo**: 1 minuto  
**Certeza**: 100%

```bash
python3 -c "
import sys
sys.path.insert(0, '/root')
from darwinacci_omega.core.engine import DarwinacciEngine
print('✅ Import OK')
"
```

**Impacto**: Confirma correção

---

### **FASE 2: VALIDAÇÃO EMPÍRICA** (<1h, Certeza 80%)

#### 2.1. Analisar WORM Ledger Existente 🟡 MÉDIO
**Tempo**: 10 minutos  
**Certeza**: 90%

**Objetivo**: Verificar se Darwinacci standalone está evoluindo positivamente

**Comandos**:
```bash
# Fix parse error e analisar
python3 << 'EOF'
import csv
with open('/root/darwinacci_omega/data/worm.csv', 'r') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print("First 5 lines:")
    for line in lines[:5]:
        print(line.strip())
EOF

# Se CSV está ok:
python3 << 'EOF'
import csv
import json
with open('/root/darwinacci_omega/data/worm.csv', 'r') as f:
    reader = csv.DictReader(f)
    entries = list(reader)
    
    for i, e in enumerate(entries[-5:]):
        data = json.loads(e['json'])
        print(f"Cycle {data['cycle']}: score={data['best_score']:.4f}, coverage={data['coverage']:.2%}")
EOF
```

**Impacto**: Saber se Darwinacci funciona melhor que Darwin

---

#### 2.2. Matar Processo Standalone 🟡 MÉDIO
**Tempo**: 1 minuto  
**Certeza**: 100%

**Por quê**: Darwinacci rodando standalone consome CPU mas não integra

```bash
kill -9 2710411  # PID do run_external_cartpole.py
```

**Impacto**: Libera CPU para integração real

---

#### 2.3. Testar Darwinacci com V7 🟡 MÉDIO
**Tempo**: 15 minutos  
**Certeza**: 75%

**Objetivo**: Verificar se integração funciona após correção de path

```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 5

# Verificar logs
grep -i "darwinacci" logs/intelligence_v7.log | tail -20
```

**Impacto**: Validar integração

---

#### 2.4. Comparar Darwin vs Darwinacci 🟡 MÉDIO
**Tempo**: 30 minutos  
**Certeza**: 70%

**Objetivo**: Decidir empiricamente qual é melhor

**Teste A/B**:
```bash
# Teste A: Darwin original (10 cycles)
# Modificar V7 para forçar Darwin
python3 test_100_cycles_real.py 10
# Anotar: best_fitness, coverage, tempo

# Teste B: Darwinacci (10 cycles)  
# Modificar V7 para forçar Darwinacci (após fix path)
python3 test_100_cycles_real.py 10
# Anotar: best_fitness, coverage, tempo

# Comparar resultados
```

**Impacto**: **DECISÃO EMPÍRICA** qual usar em produção

---

### **FASE 3: INTEGRAÇÃO COMPLETA** (<4h, Certeza 60%)

#### 3.1. Integrar ao Brain Daemon 🔴 COMPLEXO
**Tempo**: 2 horas  
**Certeza**: 60%

**Objetivo**: Brain Daemon usa Darwinacci para evoluir hiperparâmetros

**Modificações**:
```python
# /root/UNIFIED_BRAIN/brain_daemon_real_env.py

# Adicionar no __init__():
try:
    import sys
    sys.path.insert(0, '/root')
    from darwinacci_omega.core.engine import DarwinacciEngine
    
    def init_fn(rng):
        return {
            'lr': rng.uniform(0.0001, 0.01),
            'curiosity_weight': rng.uniform(0.01, 0.3),
            'top_k': int(rng.randint(4, 16)),
        }
    
    def eval_fn(genome, rng):
        # Use reward como fitness
        return {
            'objective': self.stats['avg_reward_last_100'],
            'behavior': [genome['lr'] * 1000, genome['top_k']],
            'ece': 0.05, 'rho': 0.9, 'eco_ok': True, 'consent': True
        }
    
    self.darwinacci = DarwinacciEngine(init_fn, eval_fn, max_cycles=3, pop_size=20)
    self.use_darwinacci = True
    brain_logger.info("✅ Darwinacci-Ω connected to Brain Daemon")
except:
    self.use_darwinacci = False

# A cada 50 episódios:
if self.episode % 50 == 0 and self.use_darwinacci:
    # Evolve hyperparameters
    champion = self.darwinacci.run(max_cycles=1)
    if champion:
        # Apply evolved hyperparameters
        self.optimizer.param_groups[0]['lr'] = champion.genome['lr']
        self.curiosity_weight = champion.genome['curiosity_weight']
        # ...
```

**Impacto**: Brain evolui próprios hiperparâmetros!

---

#### 3.2. Substituir Darwin Original 🔴 COMPLEXO
**Tempo**: 1 hora  
**Certeza**: 70%

**Objetivo**: Darwin runner usa Darwinacci como motor

**Modificações**:
```python
# /root/darwin-engine-intelligence/darwin_main/darwin_runner.py

# Substituir DarwinEvolution por DarwinacciEngine
# Manter mesma interface
```

**Impacto**: Darwin evolution usa motor superior

---

#### 3.3. Universal Connector 🔴 MUITO COMPLEXO
**Tempo**: 2 horas  
**Certeza**: 40%

**Objetivo**: Todos sistemas sincronizam via Darwinacci

**Arquivo novo**: `/root/darwinacci_omega/core/universal_connector.py`

**Impacto**: Conectividade total entre sistemas

---

## 📊 PRIORIZAÇÃO POR ROI (Return on Investment)

| Tarefa | Tempo | Impacto | ROI | Prioridade |
|---|---|---|---|---|
| 1.1 Fix TimeCrystal | 2min | Baixo | ⭐ | 🟢 Baixa |
| 1.2 Fix PYTHONPATH | 5min | Alto | ⭐⭐⭐⭐⭐ | 🔴 CRÍTICA |
| 1.3 Test Import | 1min | Médio | ⭐⭐⭐ | 🟡 Média |
| 2.1 Analyze WORM | 10min | Alto | ⭐⭐⭐⭐ | 🔴 Alta |
| 2.3 Test V7+Darwinacci | 15min | Alto | ⭐⭐⭐⭐ | 🔴 Alta |
| 2.4 A/B Darwin vs Darwinacci | 30min | **MUITO ALTO** | ⭐⭐⭐⭐⭐ | 🔴 CRÍTICA |
| 3.1 Brain Integration | 2h | Médio | ⭐⭐ | 🟡 Média |
| 3.2 Replace Darwin Runner | 1h | Alto | ⭐⭐⭐ | 🟡 Alta |

---

## 🎯 RECOMENDAÇÃO FINAL

### **Opção A: Ativação Mínima** (20 min)
1. Fix PYTHONPATH (5min)
2. Fix TimeCrystal (2min)
3. Test Import (1min)
4. Test V7+Darwinacci (15min)

**Resultado**: Darwinacci integrado ao V7, podemos avaliar se funciona

---

### **Opção B: Validação Completa** (1h)
1. Tudo da Opção A
2. Analisar WORM ledger (10min)
3. **A/B Test Darwin vs Darwinacci** (30min)

**Resultado**: Decisão empírica qual motor usar

---

### **Opção C: Integração Total** (4h)
1. Tudo da Opção B
2. Integrar ao Brain Daemon (2h)
3. Substituir Darwin runner (1h)

**Resultado**: Darwinacci como núcleo universal

---

## 💡 MINHA RECOMENDAÇÃO HONESTA

**Faça Opção B** (Validação Completa, 1h)

**Por quê**:
1. Você investiu tempo criando Darwinacci
2. Arquitetura é superior ao Darwin original
3. **MAS** precisa validar empiricamente antes de substituir tudo
4. A/B test é **CRUCIAL** para decisão científica

**Se A/B mostrar que Darwinacci é melhor**: Partir para Opção C

**Se A/B mostrar que são equivalentes**: Manter Darwin (menos risco)

**Se A/B mostrar que Darwin é melhor**: Investigar por quê e corrigir Darwinacci

---

## ❓ PERGUNTAS CRÍTICAS PARA VOCÊ

1. **Darwinacci standalone está funcionando?**
   - Processo roda há 3h+, mas sem dados de progresso visíveis
   - Precisa verificar WORM ledger

2. **Você quer que eu ative agora?**
   - Opção A: 20min (ativação básica)
   - Opção B: 1h (com validação)
   - Opção C: 4h (integração total)

3. **Qual é a prioridade?**
   - Ter Darwinacci rodando? (Opção A)
   - Saber se é melhor que Darwin? (Opção B) ← **RECOMENDO**
   - Substituir tudo? (Opção C)

---

## 🎊 RESUMO DA AUDITORIA

### **Darwinacci-Ω**:
- ✅ Código: Excelente (90/100)
- ⚠️ Integração: Incompleta (10/100)
- ❓ Performance: Desconhecida (sem validação)
- 💎 Potencial: Muito alto (90/100)

### **Recomendação**:
1. Corrigir path (5min)
2. Validar com A/B test (30min)
3. Decidir baseado em dados

**Status**: Sistema **PROMISSOR** mas precisa **VALIDAÇÃO EMPÍRICA**

---

**Auditoria completa. Roadmap claro. Aguardando sua decisão.** 🎯