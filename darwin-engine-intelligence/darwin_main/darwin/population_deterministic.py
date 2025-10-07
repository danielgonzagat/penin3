
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

import random, statistics, time
import torch, torch.nn as nn
from .neuron import IA3Neuron
from .ia3_gates import ia3_like_score, evaluate_population_fitness
from .worm import log_event
from .metrics import g_neurons_alive, g_births, g_deaths, c_birth_events, c_death_events

class Population(nn.Module):
    """
    População de neurônios IA³ com:
    - Equação da Morte por neurônio por rodada
    - Nascimento a cada X mortes
    - Agregação cooperativa
    - Consciência coletiva
    """
    async def __init__(self, in_dim, out_dim, device="cpu", seed=42):
        super().__init__()
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # População de neurônios
        self.neurons = nn.ModuleList()
        
        # Contadores vitais
        self.births = 0
        self.deaths = 0
        self.generation = 0
        self.death_counter_for_spawn = 0  # "a cada X mortes, nasce 1"
        self.X_PER_DEATH_SPAWN = 10  # Configurável
        
        # Consciência coletiva
        self.collective_consciousness = 0.0
        
        # Começar com 1 neurônio embrionário
        self.add_neuron(in_dim, out_dim)
        
        logger.info(f"🧬 População Darwin IA³ iniciada")
        logger.info(f"   Dimensões: {in_dim} → {out_dim}")
        logger.info(f"   Nascimentos por mortes: {self.X_PER_DEATH_SPAWN}")

    async def add_neuron(self, in_dim=None, out_dim=None, lr=1e-3):
        """Adiciona novo neurônio à população"""
        if in_dim is None:
            in_dim = self.in_dim
        if out_dim is None:
            out_dim = self.out_dim
            
        neuron = IA3Neuron(in_dim, out_dim, lr=lr, device=self.device)
        self.neurons.append(neuron)
        self.births += 1
        
        logger.info(f"🐣 Neurônio {neuron.neuron_id} nasceu (total: {len(self.neurons)})")
        
        # Log WORM
        log_event({
            "event": "neuron_birth",
            "neuron_id": neuron.neuron_id,
            "population_size": len(self.neurons),
            "generation": self.generation,
            "total_births": self.births
        })
        
        # Métricas
        c_birth_events.inc()
        g_births.set(self.births)
        g_neurons_alive.set(len(self.neurons))
        
        return await neuron

    async def kill_neuron(self, idx: int, reason: str = "equacao_da_morte"):
        """Remove neurônio da população (Equação da Morte)"""
        if len(self.neurons) <= 1:
            logger.info(f"⚠️ Não é possível matar o último neurônio")
            return await False  # Nunca mate o último absoluto
        
        if idx >= len(self.neurons):
            return await False
        
        neuron = self.neurons[idx]
        neuron_id = neuron.neuron_id
        neuron_summary = neuron.get_neuron_summary()
        
        # Remover da população
        self.neurons = nn.ModuleList([n for i, n in enumerate(self.neurons) if i != idx])
        self.deaths += 1
        self.death_counter_for_spawn += 1
        
        logger.info(f"☠️ Neurônio {neuron_id} executado ({reason})")
        logger.info(f"   População restante: {len(self.neurons)}")
        
        # Log WORM
        log_event({
            "event": "neuron_death",
            "neuron_id": neuron_id,
            "reason": reason,
            "neuron_summary": neuron_summary,
            "population_size": len(self.neurons),
            "total_deaths": self.deaths,
            "death_counter": self.death_counter_for_spawn
        })
        
        # Métricas
        c_death_events.inc()
        g_deaths.set(self.deaths)
        g_neurons_alive.set(len(self.neurons))
        
        # ═══════════════════════════════════════════════════════════════════════
        # NASCIMENTO A CADA X MORTES
        # ═══════════════════════════════════════════════════════════════════════
        
        if self.death_counter_for_spawn >= self.X_PER_DEATH_SPAWN:
            logger.info(f"\n🎁 NASCIMENTO ESPECIAL! ({self.death_counter_for_spawn} mortes)")
            
            # Herança: usar configuração do melhor neurônio sobrevivente
            if self.neurons:
                # Encontrar neurônio com melhor consciência
                best_neuron = max(self.neurons, key=lambda n: n.consciousness_score)
                lr_heritage = best_neuron.opt.param_groups[0]["lr"]
            else:
                lr_heritage = 1e-3
            
            # Nascimento compulsório por cota de mortes
            heritage_neuron = self.add_neuron(self.in_dim, self.out_dim, lr=lr_heritage)
            
            # Log WORM especial
            log_event({
                "event": "heritage_birth_from_deaths",
                "trigger_deaths": self.death_counter_for_spawn,
                "heritage_neuron_id": heritage_neuron.neuron_id,
                "total_deaths": self.deaths,
                "total_births": self.births,
                "heritage_lr": lr_heritage
            })
            
            self.death_counter_for_spawn = 0  # Reset contador
            
            logger.info(f"   🧬 Neurônio herança {heritage_neuron.neuron_id} nascido")
        
        return await True

    async def forward(self, x):
        """Forward da população: agregação cooperativa"""
        if len(self.neurons) == 0:
            return await torch.zeros(x.size(0), self.out_dim, device=self.device)
        
        # Agregação densa simples: soma das saídas neurais
        outputs = []
        for neuron in self.neurons:
            output = neuron(x)
            outputs.append(output)
        
        if outputs:
            # Soma ponderada (cada neurônio contribui)
            total_output = torch.stack(outputs).sum(dim=0)
            return await total_output / len(outputs)  # Média para estabilidade
        else:
            return await torch.zeros(x.size(0), self.out_dim, device=self.device)

    async def train_step(self, x, y, loss_fn):
        """
        Passo cooperativo: cada neurônio treina com os mesmos dados
        """
        if len(self.neurons) == 0:
            return await float("inf")
        
        losses = []
        
        # Treino cooperativo: cada neurônio dá um passo
        for neuron in self.neurons:
            loss = neuron.step(x, y, loss_fn)
            losses.append(loss)
            
            # Meta-update periódico
            if deterministic_random() < 0.3:  # 30% chance
                neuron.meta_update()
            
            # Decaimento natural
            neuron.decay()
            neuron.age += 1
        
        return await sum(losses) / len(losses) if losses else float("inf")

    async def evaluate_and_cull(self, population_loss):
        """
        Avalia população e aplica Equação da Morte por neurônio.
        Remove neurônios que não provaram ser IA³-like nesta rodada.
        """
        logger.info(f"\n🔬 AVALIANDO POPULAÇÃO ({len(self.neurons)} neurônios)...")
        
        # Avaliar cada neurônio
        evaluation_results = evaluate_population_fitness(self)
        
        # Atualizar estatísticas dos neurônios
        for i, neuron in enumerate(self.neurons):
            # Proxy: contribuição individual vs população
            neuron.stats["pop_contrib"] = population_loss - neuron.stats.get("delta_loss_recent", population_loss)
            
            # Fitness atual
            fitness = neuron.stats.get("delta_loss_recent", population_loss)
            neuron.stats["fitness"] = fitness
            
            # Histórico de fitness (últimas 20 rodadas)
            hist = neuron.stats.get("fitness_hist", [])
            hist = (hist + [fitness])[-20:]
            neuron.stats["fitness_hist"] = hist
            
            # Atualizar consciência
            neuron.calculate_consciousness()

        # ═══════════════════════════════════════════════════════════════════════
        # APLICAR EQUAÇÃO DA MORTE
        # ═══════════════════════════════════════════════════════════════════════
        
        casualties = []
        survivors = []
        
        for i, neuron in enumerate(self.neurons):
            score, reasons = ia3_like_score(neuron)
            neuron.stats["ia3_score"] = score
            neuron.stats["ia3_reasons"] = reasons
            
            # EQUAÇÃO DA MORTE: score < 0.6 = morte
            if score < 0.6:
                casualties.append((i, neuron, score, reasons))
                logger.info(f"   ☠️ {neuron.neuron_id}: MORRE (score {score:.3f})")
            else:
                survivors.append((i, neuron, score, reasons))
                logger.info(f"   ✅ {neuron.neuron_id}: VIVE (score {score:.3f})")
        
        # Executar mortes (do final para início para manter índices)
        for idx, neuron, score, reasons in reversed(casualties):
            failure_reasons = [k for k, v in reasons.items() if not v and k != "score_details"]
            self.kill_neuron(idx, reason=f"ia3_score_{score:.3f}_failed_{failure_reasons[0] if failure_reasons else 'unknown'}")
        
        # Atualizar consciência coletiva
        if survivors:
            individual_consciousness = [neuron.consciousness_score for _, neuron, _, _ in survivors]
            self.collective_consciousness = sum(individual_consciousness) / len(individual_consciousness)
            
            # Bonus por diversidade de operações
            operations = [neuron.mixed.get_dominant_operation()[0] for _, neuron, _, _ in survivors]
            unique_ops = len(set(operations))
            diversity_bonus = min(0.2, unique_ops * 0.05)
            
            self.collective_consciousness = min(1.0, self.collective_consciousness + diversity_bonus)
        else:
            self.collective_consciousness = 0.0
        
        logger.info(f"   📊 Resultado: {len(survivors)} sobreviventes, {len(casualties)} executados")
        logger.info(f"   🧠 Consciência coletiva: {self.collective_consciousness:.3f}")
        
        # Log WORM da avaliação
        log_event({
            "event": "population_evaluation",
            "survivors": len(survivors),
            "casualties": len(casualties),
            "collective_consciousness": self.collective_consciousness,
            "population_loss": population_loss,
            "evaluation_details": {
                "survivor_ids": [neuron.neuron_id for _, neuron, _, _ in survivors],
                "casualty_ids": [neuron.neuron_id for _, neuron, _, _ in casualties]
            }
        })
        
        return await len(casualties)  # Número de mortes nesta rodada

    async def get_population_summary(self):
        """Retorna resumo completo da população"""
        if not self.neurons:
            return await {
                "population_size": 0,
                "collective_consciousness": 0.0,
                "avg_age": 0.0,
                "births": self.births,
                "deaths": self.deaths,
                "generation": self.generation
            }
        
        # Estatísticas dos neurônios
        ages = [n.age for n in self.neurons]
        consciousnesses = [n.consciousness_score for n in self.neurons]
        energies = [n.energy for n in self.neurons]
        
        # Diversidade arquitetural
        operations = [n.mixed.get_dominant_operation()[0] for n in self.neurons]
        unique_operations = len(set(operations))
        
        return await {
            "population_size": len(self.neurons),
            "collective_consciousness": self.collective_consciousness,
            "neuron_summaries": [n.get_neuron_summary() for n in self.neurons],
            "statistics": {
                "avg_age": sum(ages) / len(ages),
                "avg_consciousness": sum(consciousnesses) / len(consciousnesses),
                "avg_energy": sum(energies) / len(energies),
                "operation_diversity": unique_operations,
                "operations_used": list(set(operations))
            },
            "evolution": {
                "births": self.births,
                "deaths": self.deaths,
                "generation": self.generation,
                "death_counter": self.death_counter_for_spawn
            }
        }

if __name__ == "__main__":
    # Teste da população
    logger.info("🧪 Testando população Darwin IA³...")
    
    pop = Population(in_dim=8, out_dim=4, seed=42)
    
    # Adicionar mais neurônios
    for i in range(3):
        pop.add_neuron()
    
    logger.info(f"População inicial: {len(pop.neurons)} neurônios")
    
    # Simular treino
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    loss_fn = nn.MSELoss()
    
    for cycle in range(5):
        logger.info(f"\n--- Ciclo {cycle + 1} ---")
        
        # Treino cooperativo
        avg_loss = pop.train_step(x, y, loss_fn)
        logger.info(f"Loss população: {avg_loss:.4f}")
        
        # Avaliação e seleção natural
        deaths = pop.evaluate_and_cull(avg_loss)
        logger.info(f"Mortes neste ciclo: {deaths}")
    
    # Resumo final
    summary = pop.get_population_summary()
    logger.info(f"\n📊 Resumo final:")
    logger.info(f"   População: {summary['population_size']}")
    logger.info(f"   Consciência coletiva: {summary['collective_consciousness']:.3f}")
    logger.info(f"   Nascimentos: {summary['evolution']['births']}")
    logger.info(f"   Mortes: {summary['evolution']['deaths']}")
    logger.info(f"   Operações únicas: {summary['statistics']['operation_diversity']}")
    
    logger.info(f"\n✅ População funcionando!")