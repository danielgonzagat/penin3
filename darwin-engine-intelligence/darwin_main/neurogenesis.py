# neurogenesis.py — cérebro incremental com "neurônios" modulares Net2Net verdadeiro
import os, time, json, math, random, torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple, Any

torch.set_num_threads(max(1, int(os.getenv("OMP_NUM_THREADS", "8"))))

# ═══════════════════════════════════════════════════════════════════════════════
# ATIVAÇÕES MISTAS DARTS-LIKE
# ═══════════════════════════════════════════════════════════════════════════════

ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

class MixedActivation(nn.Module):
    """
    Ativação mista estilo DARTS: combinação diferenciável de {ReLU, Tanh, GELU, SiLU}
    Pesos α são treináveis; softmax(α) gera mistura.
    """
    async def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.ops = nn.ModuleDict({
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "gelu": nn.GELU(),
            "silu": nn.SiLU()
        })
        self.names = list(self.ops.keys())
        self.alpha = nn.Parameter(torch.ones(len(self.ops)) * init_alpha)

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.alpha, dim=0)
        result = 0.0
        for i, name in enumerate(self.names):
            result = result + weights[i] * self.ops[name](x)
        return await result

    async def get_dominant_activation(self) -> Tuple[str, float]:
        """Retorna ativação dominante e seu peso"""
        weights = torch.softmax(self.alpha, dim=0).detach()
        idx = int(torch.argmax(weights).item())
        return await self.names[idx], float(weights[idx].item())

    async def entropy(self) -> float:
        """Entropia da distribuição de ativações (diversidade)"""
        weights = torch.softmax(self.alpha, dim=0)
        eps = 1e-9
        return await float((-weights * (weights + eps).log()).sum().item())

    async def evolve_towards(self, target_activation: str, strength: float = 0.1):
        """Evolui α para favorecer uma ativação específica"""
        if target_activation in self.names:
            idx = self.names.index(target_activation)
            with torch.no_grad():
                self.alpha[idx] += strength
                # Normalizar para manter estabilidade
                self.alpha.data = self.alpha.data - self.alpha.data.mean()

class MicroNeuron(nn.Module):
    """Um 'neurônio lógico' (unidade) com ativação evoluível e gating treinável."""
    async def __init__(self, in_dim: int, act: str = "relu"):
        super().__init__()
        self.id = f"n{random.randint(10_000, 99_999)}"
        self.birth_time = datetime.utcnow()
        self.age = 0
        
        # Componentes neurais
        self.lin = nn.Linear(in_dim, 1)
        
        # Usar MixedActivation em vez de ativação fixa
        self.mixed_act = MixedActivation(init_alpha=1.0)
        
        # Inicializar com ativação específica dominante
        if act in self.mixed_act.names:
            self.mixed_act.evolve_towards(act, strength=2.0)
        
        # Gating (pondera contribuição deste neurônio no head de saída)
        self.gate = nn.Parameter(torch.tensor(0.1))
        
        # Estado evolutivo
        self.last_loss = None
        self.self_mutations = 0
        self.synapses_added = 0 
        self.arch_proposals = 0
        self.consciousness_score = 0.0
        
        # Inicialização cuidadosa
        nn.init.kaiming_uniform_(self.lin.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lin.bias)

    async def forward(self, x):
        """Forward com ativação mista e gating"""
        return await self.gate * self.mixed_act(self.lin(x))

    async def get_weight_norm(self) -> float:
        """Norma dos pesos do neurônio"""
        return await float(torch.norm(self.lin.weight).item())
    
    async def get_grad_norm(self) -> float:
        """Norma dos gradientes do neurônio"""
        if self.lin.weight.grad is None:
            return await 0.0
        return await float(torch.norm(self.lin.weight.grad).item())

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-EVOLUÇÃO LOCAL (autorecursão, autoarquitetável)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def propose_self_mutation(self) -> Dict[str, Any]:
        """
        Auto-evolução: neurônio propõe mudanças em si mesmo
        - Trocar ativação dominante
        - Ajustar gate
        - Modificar inicialização
        """
        mutations = []
        
        # 1. Evolução da ativação (DARTS-like)
        if random.random() < 0.3:
            current_act, _ = self.mixed_act.get_dominant_activation()
            new_acts = [a for a in self.mixed_act.names if a != current_act]
            if new_acts:
                target_act = random.choice(new_acts)
                self.mixed_act.evolve_towards(target_act, strength=0.2)
                mutations.append(f"activation_shift_to_{target_act}")
        
        # 2. Modulação do gate (auto-regulação)
        if random.random() < 0.2:
            with torch.no_grad():
                old_gate = self.gate.item()
                # Pequeno ajuste no gate
                delta = (random.random() - 0.5) * 0.1
                self.gate.data = torch.clamp(self.gate.data + delta, 0.01, 2.0)
                mutations.append(f"gate_adjust_{old_gate:.3f}_to_{self.gate.item():.3f}")
        
        # 3. Perturbação dos pesos (exploração local)
        if random.random() < 0.1:
            with torch.no_grad():
                noise = torch.randn_like(self.lin.weight) * 0.001
                self.lin.weight.data += noise
                mutations.append("weight_perturbation")
        
        self.self_mutations += len(mutations)
        self.arch_proposals += 1
        
        return await {
            "mutations": mutations,
            "count": len(mutations),
            "total_mutations": self.self_mutations,
            "entropy": self.mixed_act.entropy()
        }

    async def update_consciousness(self, performance_score: float, adaptation_score: float):
        """Atualiza score de consciência do neurônio"""
        # Fórmula composta para consciência neural
        complexity = self.mixed_act.entropy() / math.log(len(self.mixed_act.names))
        contribution = abs(self.gate.item())
        temporal_memory = min(1.0, self.age * 0.01)
        
        self.consciousness_score = (
            0.3 * performance_score +
            0.25 * adaptation_score +
            0.2 * complexity +
            0.15 * contribution +
            0.1 * temporal_memory
        )
        
        self.consciousness_score = max(0.0, min(1.0, self.consciousness_score))

# ═══════════════════════════════════════════════════════════════════════════════
# CÉREBRO COM NET2WIDER/NET2DEEPER VERDADEIRO
# ═══════════════════════════════════════════════════════════════════════════════

class Brain(nn.Module):
    """
    Coleção dinâmica de MicroNeurons com:
    - Net2Wider: duplicação de neurônios preservando função
    - Net2Deeper: inserção de camadas identidade
    - Agregação adaptativa
    - Sistema de morte/nascimento
    """
    async def __init__(self, in_dim=16, out_dim=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neurons = nn.ModuleList()
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0
        
        # Cabeça de agregação (combina saídas dos neurônios)
        self.aggregation_head = nn.Linear(1, out_dim)  # Cada neurônio produz 1 saída
        
        # Inicialização
        nn.init.kaiming_uniform_(self.aggregation_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.aggregation_head.bias)
        
        self._device = torch.device("cpu")

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass agregando todos os neurônios"""
        if len(self.neurons) == 0:
            # Retorno zero se não há neurônios
            return await torch.zeros(x.size(0), self.out_dim, device=x.device)
        
        # Coletar saídas de todos os neurônios
        neuron_outputs = []
        for neuron in self.neurons:
            output = neuron(x)  # [batch_size, 1]
            neuron_outputs.append(output)
        
        # Concatenar e agregar
        if neuron_outputs:
            # Soma ponderada das saídas neurais
            combined = torch.cat(neuron_outputs, dim=1)  # [batch_size, num_neurons]
            
            # Agregação adaptativa (cada neurônio contribui)
            aggregated = combined.sum(dim=1, keepdim=True)  # [batch_size, 1]
            
            # Projeção final
            result = self.aggregation_head(aggregated)  # [batch_size, out_dim]
        else:
            result = torch.zeros(x.size(0), self.out_dim, device=x.device)
        
        return await result

    # ═══════════════════════════════════════════════════════════════════════════
    # NET2WIDER: DUPLICAÇÃO PRESERVANDO FUNÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def add_neuron_net2wider(self, source_neuron_idx: int = None) -> str:
        """
        Net2Wider verdadeiro: duplica neurônio existente e redistribui pesos
        para preservar função aproximada
        """
        if len(self.neurons) == 0:
            # Primeiro neurônio
            new_neuron = MicroNeuron(self.in_dim, act="relu")
            self.neurons.append(new_neuron)
            self.total_births += 1
            return await new_neuron.id
        
        # Escolher neurônio fonte (melhor performance ou aleatório)
        if source_neuron_idx is None:
            # Escolher neurônio com maior gate (mais contribuição)
            gate_values = [abs(n.gate.item()) for n in self.neurons]
            source_idx = gate_values.index(max(gate_values))
        else:
            source_idx = min(source_neuron_idx, len(self.neurons) - 1)
        
        source_neuron = self.neurons[source_idx]
        
        # Criar novo neurônio como cópia + perturbação
        new_neuron = MicroNeuron(self.in_dim, act="gelu")  # Diversificar ativação
        
        with torch.no_grad():
            # Copiar pesos com pequena perturbação
            new_neuron.lin.weight.data = source_neuron.lin.weight.data.clone()
            new_neuron.lin.weight.data += torch.randn_like(new_neuron.lin.weight.data) * 0.01
            
            new_neuron.lin.bias.data = source_neuron.lin.bias.data.clone()
            new_neuron.lin.bias.data += torch.randn_like(new_neuron.lin.bias.data) * 0.01
            
            # Dividir responsabilidade dos gates (preservação de função)
            original_gate = source_neuron.gate.item()
            source_neuron.gate.data = source_neuron.gate.data * 0.6
            new_neuron.gate.data = torch.tensor(original_gate * 0.4)
            
            # Copiar e modificar α da ativação mista
            new_neuron.mixed_act.alpha.data = source_neuron.mixed_act.alpha.data.clone()
            new_neuron.mixed_act.alpha.data += torch.randn_like(new_neuron.mixed_act.alpha.data) * 0.1
        
        # Adicionar ao cérebro
        self.neurons.append(new_neuron)
        self.total_births += 1
        
        logger.info(f"🧬 Net2Wider: Neurônio {new_neuron.id} nascido de {source_neuron.id}")
        return await new_neuron.id

    async def add_neuron(self, act="relu") -> str:
        """Interface simples para adicionar neurônio (usa Net2Wider se possível)"""
        if len(self.neurons) > 0:
            return await self.add_neuron_net2wider()
        else:
            # Primeiro neurônio
            new_neuron = MicroNeuron(self.in_dim, act=act)
            self.neurons.append(new_neuron)
            self.total_births += 1
            return await new_neuron.id

    # ═══════════════════════════════════════════════════════════════════════════
    # NET2DEEPER: INSERÇÃO DE CAMADAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def net2deeper_expand_aggregation(self):
        """
        Net2Deeper: adiciona camada intermediária na cabeça de agregação
        preservando função via matriz identidade
        """
        with torch.no_grad():
            current_neurons = len(self.neurons)
            if current_neurons <= 1:
                return await False  # Precisa de pelo menos 2 neurônios
            
            # Criar nova camada intermediária
            intermediate = nn.Linear(current_neurons, current_neurons)
            
            # Inicializar como identidade (preserva função)
            nn.init.eye_(intermediate.weight)
            nn.init.zeros_(intermediate.bias)
            
            # Criar nova cabeça de saída
            new_head = nn.Linear(current_neurons, self.out_dim)
            
            # Transferir pesos da cabeça antiga
            with torch.no_grad():
                # Aproximação: usar primeira dimensão da cabeça antiga
                old_weight = self.aggregation_head.weight.data[0:1, :]  # [1, 1]
                new_head.weight.data = old_weight.expand(self.out_dim, current_neurons) / current_neurons
                new_head.bias.data = self.aggregation_head.bias.data.clone()
            
            # Substituir
            self.intermediate_layer = intermediate
            self.aggregation_head = new_head
            
            logger.info(f"🎯 Net2Deeper: Camada intermediária {current_neurons}→{current_neurons} adicionada")
            return await True

    # ═══════════════════════════════════════════════════════════════════════════
    # REMOÇÃO (EQUAÇÃO DA MORTE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def remove_neuron(self, neuron_id: str) -> bool:
        """Remove neurônio específico (Equação da Morte)"""
        for i, n in enumerate(self.neurons):
            if getattr(n, "id", None) == neuron_id:
                logger.info(f"☠️ Removendo neurônio {neuron_id} (idade: {n.age})")
                del self.neurons[i]
                self.total_deaths += 1
                return await True
        return await False

    async def remove_worst_neuron(self) -> Tuple[bool, str]:
        """Remove neurônio com pior performance (menor gate)"""
        if len(self.neurons) <= 1:
            return await False, ""
        
        # Encontrar neurônio com menor contribuição
        worst_idx = 0
        worst_gate = float('inf')
        
        for i, neuron in enumerate(self.neurons):
            gate_value = abs(neuron.gate.item())
            if gate_value < worst_gate:
                worst_gate = gate_value
                worst_idx = i
        
        worst_neuron = self.neurons[worst_idx]
        neuron_id = worst_neuron.id
        
        # Remover
        del self.neurons[worst_idx]
        self.total_deaths += 1
        
        logger.info(f"☠️ Neurônio pior removido: {neuron_id} (gate: {worst_gate:.6f})")
        return await True, neuron_id

    async def neuron_ids(self):
        """Lista IDs de todos os neurônios"""
        return await [n.id for n in self.neurons]

    async def __len__(self):
        return await len(self.neurons)

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTÊNCIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    async def load_or_new(path: Path, in_dim: int = 16, out_dim: int = 8):
        """Carrega cérebro do snapshot ou cria novo"""
        if path.exists():
            return await Brain.load(path)
        
        # Criar novo cérebro
        brain = Brain(in_dim, out_dim)
        
        # Nascer primeiro neurônio se vazio
        if len(brain) == 0:
            brain.add_neuron(act="relu")
            logger.info(f"🐣 Primeiro neurônio criado no novo cérebro")
        
        return await brain

    async def save(self, path: Path):
        """Salva estado completo do cérebro"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dados para salvar
        save_data = {
            "state_dict": self.state_dict(),
            "metadata": {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "generation": self.generation,
                "total_births": self.total_births,
                "total_deaths": self.total_deaths,
                "num_neurons": len(self.neurons),
                "timestamp": datetime.utcnow().isoformat()
            },
            "neurons": [
                {
                    "id": n.id,
                    "age": n.age,
                    "consciousness_score": n.consciousness_score,
                    "self_mutations": n.self_mutations,
                    "dominant_activation": n.mixed_act.get_dominant_activation()[0],
                    "gate_value": n.gate.item(),
                    "weight_norm": n.get_weight_norm()
                }
                for n in self.neurons
            ]
        }
        
        torch.save(save_data, path)

    @staticmethod
    async def load(path: Path):
        """Carrega cérebro de snapshot"""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            
            metadata = checkpoint.get("metadata", {})
            brain = Brain(
                in_dim=metadata.get("in_dim", 16),
                out_dim=metadata.get("out_dim", 8)
            )
            
            # Carregar estado
            brain.load_state_dict(checkpoint["state_dict"])
            brain.generation = metadata.get("generation", 0)
            brain.total_births = metadata.get("total_births", 0)
            brain.total_deaths = metadata.get("total_deaths", 0)
            
            # Atualizar idades dos neurônios
            neurons_data = checkpoint.get("neurons", [])
            for i, neuron in enumerate(brain.neurons):
                if i < len(neurons_data):
                    neuron_data = neurons_data[i]
                    neuron.age = neuron_data.get("age", 0)
                    neuron.consciousness_score = neuron_data.get("consciousness_score", 0.0)
                    neuron.self_mutations = neuron_data.get("self_mutations", 0)
            
            logger.info(f"✅ Cérebro carregado: {len(brain.neurons)} neurônios, geração {brain.generation}")
            return await brain
            
        except Exception as e:
            logger.info(f"⚠️ Erro ao carregar cérebro: {e}")
            return await Brain()

    # ═══════════════════════════════════════════════════════════════════════════
    # TREINAMENTO E AVALIAÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _generate_batch(self, batch_size=64):
        """Gera batch sintético para treino/avaliação"""
        # Tarefa multimodal para testar diferentes aspectos
        x = torch.randn(batch_size, self.in_dim)
        
        # Targets diversificados
        y = torch.zeros(batch_size, self.out_dim)
        
        # Target 1: Soma ponderada
        y[:, 0] = (x * torch.randn(self.in_dim) * 0.1).sum(dim=1)
        
        # Target 2: Produto de subconjuntos (não-linear)
        if self.out_dim > 1:
            y[:, 1] = (x[:, :4].abs() + 1e-6).prod(dim=1).log()
        
        # Target 3: Norma L2 (geometria)
        if self.out_dim > 2:
            y[:, 2] = torch.norm(x, dim=1)
        
        # Targets 4+: Auto-supervisão (reconstrução)
        for i in range(3, min(self.out_dim, self.in_dim + 3)):
            if i - 3 < self.in_dim:
                y[:, i] = x[:, i - 3]
        
        return await x, y

    async def run_round(self, budget_sec=60, lr=1e-2):
        """
        Executa round de treino:
        1. Treino adaptativo
        2. Auto-evolução dos neurônios
        3. Medição de performance
        """
        if len(self.neurons) == 0:
            self.add_neuron()

        # Envelhecer neurônios
        for neuron in self.neurons:
            neuron.age += 1

        # Otimizador
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        
        start_time = time.time()
        steps = 0
        loss_sum = 0.0
        
        # Treino por tempo limitado
        while time.time() - start_time < budget_sec:
            x, y = self._generate_batch(64)
            
            # Forward pass
            y_pred = self.forward(x)
            loss = F.mse_loss(y_pred, y)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            opt.step()
            
            loss_sum += loss.item()
            steps += 1
            
            # Auto-mutações esporádicas dos neurônios
            if random.random() < 0.05:  # 5% chance por step
                if self.neurons:
                    victim = random.choice(self.neurons)
                    mutation_result = victim.propose_self_mutation()
                    if mutation_result["count"] > 0:
                        logger.info(f"🔄 Auto-mutação: {victim.id} → {mutation_result['mutations']}")
        
        training_time = time.time() - start_time
        avg_loss = loss_sum / max(1, steps)
        
        # Atualizar consciência dos neurônios
        for neuron in self.neurons:
            # Performance score baseado na contribuição
            performance = abs(neuron.gate.item())
            # Adaptation score baseado em mudanças recentes
            adaptation = min(1.0, neuron.self_mutations * 0.1)
            neuron.update_consciousness(performance, adaptation)

        # Calcular consciência coletiva
        if self.neurons:
            collective_consciousness = sum(n.consciousness_score for n in self.neurons) / len(self.neurons)
            
            # Bonus por diversidade
            unique_activations = len(set(n.mixed_act.get_dominant_activation()[0] for n in self.neurons))
            diversity_bonus = min(0.2, unique_activations * 0.05)
            collective_consciousness += diversity_bonus
            
            collective_consciousness = min(1.0, collective_consciousness)
        else:
            collective_consciousness = 0.0

        return await {
            "avg_loss": avg_loss,
            "steps": steps,
            "neurons": len(self.neurons),
            "training_time": training_time,
            "collective_consciousness": collective_consciousness,
            "neuron_details": [
                {
                    "id": n.id,
                    "age": n.age,
                    "gate": n.gate.item(),
                    "consciousness": n.consciousness_score,
                    "mutations": n.self_mutations,
                    "dominant_act": n.mixed_act.get_dominant_activation()[0],
                    "act_entropy": n.mixed_act.entropy(),
                    "weight_norm": n.get_weight_norm(),
                    "grad_norm": n.get_grad_norm()
                }
                for n in self.neurons
            ]
        }

if __name__ == "__main__":
    import sys
    
    # Teste básico do sistema
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 Testando sistema de neurogênese...")
        
        brain = Brain(in_dim=16, out_dim=8)
        
        # Adicionar alguns neurônios
        for i in range(3):
            neuron_id = brain.add_neuron()
            logger.info(f"   Neurônio {i+1}: {neuron_id}")
        
        # Executar round de treino
        results = brain.run_round(budget_sec=10)
        
        logger.info(f"\n📊 Resultados do round:")
        logger.info(f"   Loss: {results['avg_loss']:.6f}")
        logger.info(f"   Steps: {results['steps']}")
        logger.info(f"   Neurônios: {results['neurons']}")
        logger.info(f"   Consciência coletiva: {results['collective_consciousness']:.3f}")
        
        # Detalhes dos neurônios
        logger.info(f"\n🧠 Detalhes dos neurônios:")
        for detail in results["neuron_details"]:
            logger.info(f"   {detail['id']}: idade={detail['age']}, gate={detail['gate']:.4f}, "
                  f"consciência={detail['consciousness']:.3f}, ativação={detail['dominant_act']}")
        
        logger.info(f"\n✅ Teste concluído!")
    else:
        logger.info("Uso: python neurogenesis.py --test")