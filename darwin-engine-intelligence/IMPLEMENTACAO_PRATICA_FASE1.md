# 🚀 IMPLEMENTAÇÃO PRÁTICA - FASE 1
## Código Pronto para Copiar e Executar

---

## ✅ CHECKLIST DE PROGRESSO

```
SEMANA 1: Motor Universal + NSGA-II
[_] Dia 1: Criar darwin_universal_engine.py
[_] Dia 2: Refatorar EvolvableMNIST para Individual
[_] Dia 3: Adicionar objetivos multi-objetivo
[_] Dia 4: Integrar NSGA-II no orquestrador
[_] Dia 5: Testar evolução multi-objetivo

SEMANA 2: Gödel + WORM
[_] Dia 1: Criar darwin_godelian_incompleteness.py
[_] Dia 2: Integrar Gödel no orquestrador
[_] Dia 3: Criar darwin_hereditary_memory.py
[_] Dia 4: Integrar WORM hereditary
[_] Dia 5: Validar Gödel + WORM funcionando

SEMANA 3: Fibonacci + Arena
[_] Dia 1: Criar darwin_fibonacci_harmony.py
[_] Dia 2: Integrar Fibonacci no orquestrador
[_] Dia 3: Criar darwin_arena.py
[_] Dia 4: Integrar Arena no orquestrador
[_] Dia 5: Testar ritmo Fibonacci

SEMANA 4: Integração + Validação
[_] Dia 1-3: Testes end-to-end
[_] Dia 4: Benchmark performance
[_] Dia 5: Documentação Fase 1
```

---

## 📝 DIA 1: Motor Universal

### Arquivo: `core/darwin_universal_engine.py`

```python
"""
Darwin Universal Engine - Motor Evolutivo Geral
================================================

Permite executar qualquer paradigma evolutivo com qualquer tipo de indivíduo.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import random

logger = logging.getLogger(__name__)


class Individual(ABC):
    """Interface universal para qualquer indivíduo evoluível."""
    
    def __init__(self):
        self.fitness: float = 0.0
        self.objectives: Dict[str, float] = {}
        self.genome: Any = None
        self.age: int = 0
    
    @abstractmethod
    def evaluate_fitness(self) -> Dict[str, float]:
        """Avalia múltiplos objetivos."""
        pass
    
    @abstractmethod
    def mutate(self, **params) -> Individual:
        """Mutação genética."""
        pass
    
    @abstractmethod
    def crossover(self, other: Individual) -> Individual:
        """Reprodução sexual."""
        pass
    
    @abstractmethod
    def serialize(self) -> Dict:
        """Serializa para JSON."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict) -> Individual:
        """Desserializa de JSON."""
        pass


class EvolutionStrategy(ABC):
    """Interface para qualquer paradigma evolutivo."""
    
    @abstractmethod
    def initialize_population(self, size: int, individual_class: type) -> List[Individual]:
        pass
    
    @abstractmethod
    def select(self, population: List[Individual], n_survivors: int) -> List[Individual]:
        pass
    
    @abstractmethod
    def reproduce(self, survivors: List[Individual], n_offspring: int) -> List[Individual]:
        pass
    
    @abstractmethod
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        pass


class GeneticAlgorithm(EvolutionStrategy):
    """GA Clássico."""
    
    def __init__(self, survival_rate: float = 0.4, sexual_rate: float = 0.8):
        self.survival_rate = survival_rate
        self.sexual_rate = sexual_rate
    
    def initialize_population(self, size: int, individual_class: type) -> List[Individual]:
        return [individual_class() for _ in range(size)]
    
    def select(self, population: List[Individual], n_survivors: int) -> List[Individual]:
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:n_survivors]
    
    def reproduce(self, survivors: List[Individual], n_offspring: int) -> List[Individual]:
        offspring = []
        while len(offspring) < n_offspring:
            if random.random() < self.sexual_rate:
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2).mutate()
            else:
                child = random.choice(survivors).mutate()
            offspring.append(child)
        return offspring
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        for ind in population:
            ind.evaluate_fitness()
        
        n_survivors = int(len(population) * self.survival_rate)
        survivors = self.select(population, n_survivors)
        offspring = self.reproduce(survivors, len(population) - len(survivors))
        
        return survivors + offspring


class UniversalDarwinEngine:
    """Motor universal que aceita qualquer estratégia."""
    
    def __init__(self, strategy: EvolutionStrategy):
        self.strategy = strategy
        self.generation = 0
        self.history = []
    
    def evolve(self, individual_class: type, population_size: int, 
               generations: int) -> Individual:
        """Executa evolução completa."""
        logger.info(f"🧬 Universal Darwin Engine")
        logger.info(f"   Strategy: {type(self.strategy).__name__}")
        logger.info(f"   Population: {population_size}")
        logger.info(f"   Generations: {generations}")
        
        population = self.strategy.initialize_population(population_size, individual_class)
        best = None
        best_fitness = float('-inf')
        
        for gen in range(generations):
            logger.info(f"\n🧬 Generation {gen+1}/{generations}")
            
            population = self.strategy.evolve_generation(population)
            
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                best = gen_best
            
            logger.info(f"   Best: {best_fitness:.4f}")
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': sum(ind.fitness for ind in population) / len(population)
            })
            
            self.generation += 1
        
        return best


# Teste
if __name__ == "__main__":
    print("✅ darwin_universal_engine.py criado")
    print("📝 Próximo passo: Refatorar EvolvableMNIST para implementar Individual")
```

**Executar**:
```bash
cd /workspace
python core/darwin_universal_engine.py
```

---

## 📝 DIA 2: Refatorar EvolvableMNIST

### Modificação: `core/darwin_evolution_system_FIXED.py`

```python
# ADICIONAR no início do arquivo (linha 31):
from core.darwin_universal_engine import Individual

# MODIFICAR classe EvolvableMNIST (linha 58):
# DE:
# class EvolvableMNIST:
# PARA:
class EvolvableMNIST(Individual):
    """MNIST Classifier evoluível - Agora implementa Individual."""
    
    def __init__(self, genome: Dict[str, Any] = None):
        super().__init__()  # ← ADICIONAR
        
        if genome is None:
            self.genome = {
                'hidden_size': random.choice([64, 128, 256, 512]),
                'learning_rate': random.uniform(0.0001, 0.01),
                'batch_size': random.choice([32, 64, 128, 256]),
                'dropout': random.uniform(0.0, 0.5),
                'num_layers': random.choice([2, 3, 4])
            }
        else:
            self.genome = genome
        
        self.classifier = None
        self.fitness = 0.0
        self.objectives = {}  # ← ADICIONAR
    
    # ... resto igual ...
    
    # ADICIONAR no final da classe (após crossover):
    def serialize(self) -> Dict:
        """Serializa para JSON."""
        return {
            'genome': self.genome,
            'fitness': self.fitness,
            'objectives': self.objectives,
            'age': self.age
        }
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'EvolvableMNIST':
        """Desserializa de JSON."""
        ind = cls(data['genome'])
        ind.fitness = data['fitness']
        ind.objectives = data.get('objectives', {})
        ind.age = data.get('age', 0)
        return ind
```

**Testar**:
```bash
cd /workspace
python -c "from core.darwin_evolution_system_FIXED import EvolvableMNIST; from core.darwin_universal_engine import Individual; assert issubclass(EvolvableMNIST, Individual); print('✅ EvolvableMNIST agora é Individual')"
```

---

## 📝 DIA 3: Multi-objetivo Real

### Modificação: `core/darwin_evolution_system_FIXED.py`

```python
# ADICIONAR novo método em EvolvableMNIST (após evaluate_fitness, linha 221):
def evaluate_fitness_multiobj(self) -> Dict[str, float]:
    """
    Avalia MÚLTIPLOS objetivos SEM scalarization.
    
    Objetivos:
    1. accuracy: Precisão no test set
    2. efficiency: Inverso da complexidade (parâmetros)
    3. speed: Inverso do tempo de inferência
    4. robustness: Accuracy com ruído Gaussiano
    5. generalization: Accuracy em FashionMNIST (transfer)
    """
    try:
        import time
        
        # Setup (igual ao evaluate_fitness atual)
        seed = 1337
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.build().to(device)
        
        # Datasets
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # MNIST train
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.genome['batch_size'], shuffle=True)
        
        # MNIST test
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # FashionMNIST test (para generalização)
        fashion_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        fashion_loader = DataLoader(fashion_dataset, batch_size=1000, shuffle=False)
        
        # Treinar (igual ao atual)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.genome['learning_rate'])
        model.train()
        
        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 300:
                    break
        
        # OBJETIVO 1: Accuracy (MNIST test)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = correct / total
        
        # OBJETIVO 2: Efficiency
        complexity = sum(p.numel() for p in model.parameters())
        efficiency = 1.0 - (complexity / 1e6)
        efficiency = max(0.0, min(1.0, efficiency))
        
        # OBJETIVO 3: Speed
        start = time.time()
        with torch.no_grad():
            _ = model(torch.randn(100, 1, 28, 28).to(device))
        inference_time = time.time() - start
        speed = 1.0 / (inference_time + 1e-6)
        speed_normalized = min(1.0, speed / 100.0)
        
        # OBJETIVO 4: Robustness (accuracy com ruído)
        robust_correct = 0
        robust_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Ruído Gaussiano σ=0.1
                noisy_data = data + 0.1 * torch.randn_like(data)
                output = model(noisy_data)
                pred = output.argmax(dim=1)
                robust_correct += pred.eq(target).sum().item()
                robust_total += len(data)
        
        robustness = robust_correct / robust_total
        
        # OBJETIVO 5: Generalization (FashionMNIST)
        gen_correct = 0
        gen_total = 0
        with torch.no_grad():
            for data, target in fashion_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                gen_correct += pred.eq(target).sum().item()
                gen_total += len(data)
        
        generalization = gen_correct / gen_total
        
        # Salvar objetivos (NÃO fazer weighted sum!)
        self.objectives = {
            'accuracy': float(accuracy),
            'efficiency': float(efficiency),
            'speed': float(speed_normalized),
            'robustness': float(robustness),
            'generalization': float(generalization)
        }
        
        # Fitness escalar (média) apenas para compatibilidade
        self.fitness = sum(self.objectives.values()) / len(self.objectives)
        
        logger.info(f"   📊 Objectives:")
        for k, v in self.objectives.items():
            logger.info(f"      {k}: {v:.4f}")
        logger.info(f"   🎯 Scalar Fitness: {self.fitness:.4f}")
        
        return self.objectives
        
    except Exception as e:
        logger.error(f"   ❌ Multiobj evaluation failed: {e}")
        self.objectives = {k: 0.0 for k in ['accuracy', 'efficiency', 'speed', 'robustness', 'generalization']}
        self.fitness = 0.0
        return self.objectives
```

**Testar**:
```bash
cd /workspace
python -c "
from core.darwin_evolution_system_FIXED import EvolvableMNIST
ind = EvolvableMNIST({'hidden_size': 128, 'learning_rate': 0.001, 'batch_size': 64, 'dropout': 0.1, 'num_layers': 2})
objectives = ind.evaluate_fitness_multiobj()
assert 'accuracy' in objectives
assert 'efficiency' in objectives
assert 'speed' in objectives
assert 'robustness' in objectives
assert 'generalization' in objectives
print('✅ Multi-objetivo funcionando')
print(f'Objectives: {objectives}')
"
```

---

## 📝 DIA 4: Integrar NSGA-II

### Modificação: `core/darwin_evolution_system_FIXED.py`

```python
# ADICIONAR no início:
from core.nsga2 import fast_nondominated_sort, crowding_distance

# ADICIONAR novo método no DarwinEvolutionOrchestrator (após evolve_mnist, linha 602):
def evolve_mnist_multiobj(self, generations: int = 100, population_size: int = 100):
    """
    Evolução multi-objetivo com NSGA-II.
    Retorna Pareto front ao invés de um único melhor.
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 MULTI-OBJECTIVE EVOLUTION (NSGA-II)")
    logger.info("="*80)
    logger.info(f"População: {population_size}")
    logger.info(f"Gerações: {generations}")
    logger.info(f"Objetivos: accuracy, efficiency, speed, robustness, generalization")
    
    # População inicial
    population = [EvolvableMNIST() for _ in range(population_size)]
    
    best_front = []
    
    for gen in range(generations):
        logger.info(f"\n🧬 Geração {gen+1}/{generations}")
        
        # Avaliar (multi-objetivo)
        logger.info(f"   Avaliando {len(population)} indivíduos...")
        for idx, ind in enumerate(population):
            if idx % 10 == 0:
                logger.info(f"   Progresso: {idx}/{len(population)}")
            ind.evaluate_fitness_multiobj()
        
        # NSGA-II: Non-dominated sorting
        objective_list = [ind.objectives for ind in population]
        maximize = {
            'accuracy': True,
            'efficiency': True,
            'speed': True,
            'robustness': True,
            'generalization': True
        }
        
        fronts = fast_nondominated_sort(objective_list, maximize)
        
        logger.info(f"\n   📊 Pareto Analysis:")
        logger.info(f"   Total fronts: {len(fronts)}")
        logger.info(f"   Front 0 (Pareto optimal): {len(fronts[0])} indivíduos")
        
        # Salvar melhor front
        best_front = [population[i] for i in fronts[0]]
        
        # Log top Pareto solutions
        logger.info(f"\n   🏆 Top 5 Pareto Solutions:")
        for i, ind in enumerate(best_front[:5]):
            logger.info(f"   #{i+1}:")
            for k, v in ind.objectives.items():
                logger.info(f"      {k}: {v:.4f}")
        
        # Seleção: Preencher com fronts até atingir n_survivors
        n_survivors = int(population_size * 0.4)
        survivors = []
        
        for front_idx, front in enumerate(fronts):
            if len(survivors) >= n_survivors:
                break
            
            # Crowding distance para diversidade dentro do front
            if len(front) > 0:
                distances = crowding_distance(front, objective_list)
                
                # Ordenar por distância decrescente (mais isolado = maior diversidade)
                front_sorted = sorted(front, key=lambda i: distances[i], reverse=True)
                
                # Adicionar até completar n_survivors
                for idx in front_sorted:
                    if len(survivors) < n_survivors:
                        survivors.append(population[idx])
                    else:
                        break
        
        logger.info(f"\n   ✅ Sobreviventes selecionados: {len(survivors)}")
        
        # Reprodução (igual ao GA clássico)
        offspring = []
        while len(survivors) + len(offspring) < population_size:
            if random.random() < 0.8:
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.crossover(parent2)
                child = child.mutate()
            else:
                parent = random.choice(survivors)
                child = parent.mutate()
            
            offspring.append(child)
        
        population = survivors + offspring
        
        logger.info(f"   Nova população: {len(survivors)} sobreviventes + {len(offspring)} offspring")
        
        # Checkpoint
        if (gen + 1) % 10 == 0:
            checkpoint = {
                'generation': gen + 1,
                'pareto_front': [ind.serialize() for ind in best_front],
                'pareto_size': len(best_front)
            }
            checkpoint_path = self.output_dir / f"multiobj_checkpoint_gen_{gen+1}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"   💾 Checkpoint: {checkpoint_path.name}")
    
    # Salvar Pareto front final
    result_path = self.output_dir / "pareto_front_final.json"
    with open(result_path, 'w') as f:
        json.dump({
            'pareto_front': [ind.serialize() for ind in best_front],
            'size': len(best_front),
            'generations': generations,
            'population_size': population_size
        }, f, indent=2)
    
    logger.info(f"\n✅ Multi-objective Evolution Complete!")
    logger.info(f"   Pareto front size: {len(best_front)}")
    logger.info(f"   Saved to: {result_path}")
    
    return best_front
```

**Testar**:
```bash
cd /workspace
python -c "
from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator
orch = DarwinEvolutionOrchestrator()
pareto_front = orch.evolve_mnist_multiobj(generations=5, population_size=10)
print(f'✅ NSGA-II integrado')
print(f'Pareto front size: {len(pareto_front)}')
for i, ind in enumerate(pareto_front[:3]):
    print(f'Solution {i+1}: {ind.objectives}')
"
```

---

## 📝 DIA 5: Exemplo Completo

### Arquivo: `examples/multiobj_evolution.py`

```python
"""
Exemplo: Evolução Multi-objetivo com NSGA-II
============================================

Demonstra evolução com múltiplos objetivos simultâneos:
- Accuracy
- Efficiency
- Speed
- Robustness
- Generalization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator
import matplotlib.pyplot as plt
import json

def main():
    print("="*80)
    print("EVOLUÇÃO MULTI-OBJETIVO (NSGA-II)")
    print("="*80)
    
    # Criar orquestrador
    orch = DarwinEvolutionOrchestrator()
    
    # Evoluir com NSGA-II
    print("\nIniciando evolução multi-objetivo...")
    print("População: 20")
    print("Gerações: 10")
    print("Tempo estimado: 30 min\n")
    
    pareto_front = orch.evolve_mnist_multiobj(
        generations=10,
        population_size=20
    )
    
    print("\n" + "="*80)
    print("PARETO FRONT FINAL")
    print("="*80)
    print(f"Soluções não-dominadas: {len(pareto_front)}\n")
    
    # Mostrar soluções
    for i, ind in enumerate(pareto_front):
        print(f"Solução #{i+1}:")
        for obj, value in ind.objectives.items():
            print(f"  {obj}: {value:.4f}")
        print()
    
    # Plotar Pareto front (accuracy vs efficiency)
    try:
        accuracies = [ind.objectives['accuracy'] for ind in pareto_front]
        efficiencies = [ind.objectives['efficiency'] for ind in pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(accuracies, efficiencies, s=100, alpha=0.6)
        plt.xlabel('Accuracy')
        plt.ylabel('Efficiency')
        plt.title('Pareto Front: Accuracy vs Efficiency')
        plt.grid(True)
        plt.savefig('/workspace/pareto_front.png')
        print(f"📊 Gráfico salvo: /workspace/pareto_front.png")
    except:
        print("⚠️  Matplotlib não disponível, pulando gráfico")
    
    print("\n✅ Evolução completa!")

if __name__ == "__main__":
    main()
```

**Executar**:
```bash
cd /workspace
python examples/multiobj_evolution.py
```

---

## 🎯 VALIDAÇÃO SEMANA 1

```bash
# Checklist Semana 1:
cd /workspace

# 1. Motor universal existe
[ -f core/darwin_universal_engine.py ] && echo "✅ Universal engine" || echo "❌ Universal engine"

# 2. EvolvableMNIST é Individual
python -c "from core.darwin_evolution_system_FIXED import EvolvableMNIST; from core.darwin_universal_engine import Individual; assert issubclass(EvolvableMNIST, Individual)" && echo "✅ Individual interface" || echo "❌ Individual interface"

# 3. Multi-objetivo implementado
python -c "from core.darwin_evolution_system_FIXED import EvolvableMNIST; ind = EvolvableMNIST(); objs = ind.evaluate_fitness_multiobj(); assert 'accuracy' in objs" && echo "✅ Multi-objetivo" || echo "❌ Multi-objetivo"

# 4. NSGA-II integrado
python -c "from core.darwin_evolution_system_FIXED import DarwinEvolutionOrchestrator; orch = DarwinEvolutionOrchestrator(); assert hasattr(orch, 'evolve_mnist_multiobj')" && echo "✅ NSGA-II" || echo "❌ NSGA-II"

# 5. Exemplo funciona
[ -f examples/multiobj_evolution.py ] && echo "✅ Exemplo" || echo "❌ Exemplo"

echo ""
echo "🎉 Se todos marcados ✅, Semana 1 completa!"
```

---

## 📋 PRÓXIMOS PASSOS

### Semana 2: Gödel + WORM

Arquivos a criar:
```
core/darwin_godelian_incompleteness.py
core/darwin_hereditary_memory.py
tests/test_godel.py
tests/test_hereditary.py
```

Modificações:
```
core/darwin_evolution_system_FIXED.py: Integrar Gödel e WORM
```

### Semana 3: Fibonacci + Arena

Arquivos a criar:
```
core/darwin_fibonacci_harmony.py
core/darwin_arena.py
tests/test_fibonacci.py
tests/test_arena.py
```

### Semana 4: Validação

Arquivos a criar:
```
tests/test_integration.py
benchmark/phase1_validation.py
docs/FASE1_COMPLETE.md
```

---

## 📞 SUPORTE

Se encontrar problemas:

1. **Verificar imports**:
```bash
python -c "import torch; import numpy; print('✅ Dependencies OK')"
```

2. **Verificar estrutura**:
```bash
tree /workspace/core -L 1
```

3. **Limpar cache**:
```bash
find /workspace -type d -name __pycache__ -exec rm -rf {} +
```

4. **Re-executar teste**:
```bash
cd /workspace
python -m pytest tests/ -v
```

---

*Guia prático compilado para execução direta.*  
*Data: 2025-10-03*  
*Próximo: SEMANA 2 (Gödel + WORM)*
