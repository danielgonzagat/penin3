# /root/darwin/neurons/ia3_neuron.py
# Célula-neurônio IA³ real (PyTorch): auto-mutação, auto-evolução, prova de vida

import math, random, json, os, time
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO E CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

@dataclass
class TrainConfig:
    input_dim: int = 16
    hidden_dim: int = 16
    lr: float = 1e-2
    steps: int = 200
    seed: int = 42
    batch: int = 64
    device: str = "cpu"
    act: str = "relu"

# ═══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DE DADOS SINTÉTICOS
# ═══════════════════════════════════════════════════════════════════════════════

async def _make_data(n=512, d_in=16, seed=0, complexity_level=1):
    """
    Gera dados sintéticos com diferentes níveis de complexidade
    para testar capacidades adaptativas do neurônio
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    
    if complexity_level == 1:
        # Nível 1: Função linear com ruído
        weights = torch.randn(d_in, generator=g) * 0.1
        y = (X * weights).sum(dim=1, keepdim=True)
        y += 0.05 * torch.randn(n, 1, generator=g)
    
    elif complexity_level == 2:
        # Nível 2: Função não-linear suave
        y = (X[:, :4].sum(dim=1) + 0.5*torch.sin(X[:, 4:8]).sum(dim=1)).unsqueeze(1)
        y = (y - y.mean()) / (y.std() + 1e-6)
    
    else:
        # Nível 3: Função complexa multimodal
        y1 = torch.sum(X[:, :4], dim=1)
        y2 = torch.prod(X[:, 4:8].abs() + 1e-6, dim=1).log()
        y3 = torch.norm(X[:, 8:12], dim=1)
        y = torch.stack([y1, y2, y3], dim=1).mean(dim=1, keepdim=True)
        y = (y - y.mean()) / (y.std() + 1e-6)
    
    # Split treino/validação
    n_train = int(0.8 * n)
    return await (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])

async def _make_ood_data(base_X, base_y, shift_factor=0.2):
    """Cria dados Out-of-Distribution para teste de adaptação"""
    # Shift na distribuição
    ood_X = base_X + shift_factor * torch.randn_like(base_X)
    
    # Rotação leve
    if base_X.size(1) >= 2:
        angle = shift_factor * 0.1
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x0, x1 = ood_X[:, 0].clone(), ood_X[:, 1].clone()
        ood_X[:, 0] = cos_a * x0 - sin_a * x1
        ood_X[:, 1] = sin_a * x0 + cos_a * x1
    
    # Target levemente ajustado
    ood_y = base_y + shift_factor * 0.1 * torch.randn_like(base_y)
    
    return await ood_X, ood_y

# ═══════════════════════════════════════════════════════════════════════════════
# CÉLULA-NEURÔNIO IA³ AVANÇADA
# ═══════════════════════════════════════════════════════════════════════════════

class IA3Neuron(nn.Module):
    """
    Um 'neurônio-célula' IA³-like: módulo mínimo capaz de:
    - aprender (backprop)
    - se auto-mutacionar (trocar ativação / inserir gate / perturbar pesos)
    - provar melhora em validação (delta_loss<0)
    - produzir 'novelty' (mudança paramétrica/estrutural)
    - calcular própria consciência
    """
    async def __init__(self, cfg: TrainConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        
        self.cfg = cfg
        self.neuron_id = f"n{random.randint(10_000, 99_999)}"
        self.birth_time = datetime.utcnow()
        self.age = 0
        self.generation = 0
        
        # Arquitetura inicial: neurônio simples
        self.fc = nn.Linear(cfg.input_dim, 1)
        self.gate = nn.Identity()  # Pode evoluir para nn.Linear(1,1)
        self.act = ACTIVATIONS.get(cfg.act, nn.ReLU)()
        
        # Estado evolutivo
        self.evolution_history = []
        self.mutation_count = 0
        self.successful_mutations = 0
        self.consciousness_score = 0.0
        
        # Snapshot dos pesos para calcular novidade
        self.register_buffer("last_weight_snapshot", torch.zeros_like(self.fc.weight))
        self.last_weight_snapshot.copy_(self.fc.weight.detach())
        
        # Inicialização cuidadosa
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc.bias)

    async def forward(self, x):
        """Forward pass: fc → gate → ativação"""
        z = self.fc(x)
        z = self.gate(z)
        return await self.act(z)

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-MUTAÇÃO E AUTO-EVOLUÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def propose_self_mutation(self) -> Dict[str, Any]:
        """
        Auto-evolução: neurônio propõe mudanças em si mesmo
        - Trocar ativação
        - Evoluir gate (Identity → Linear learnable)
        - Perturbação de pesos direcionada
        """
        mutations_applied = []
        
        # Mutação 1: Troca de ativação (25% chance)
        if random.random() < 0.25:
            current_act = self.cfg.act
            available_acts = [a for a in ACTIVATIONS.keys() if a != current_act]
            if available_acts:
                new_act = random.choice(available_acts)
                self.act = ACTIVATIONS[new_act]()
                self.cfg.act = new_act
                mutations_applied.append(f"activation_change_{current_act}_to_{new_act}")
        
        # Mutação 2: Evolução do gate (15% chance)
        if random.random() < 0.15 and isinstance(self.gate, nn.Identity):
            self.gate = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.gate.weight, 1.0)  # Iniciar como identidade
            mutations_applied.append("gate_evolution_identity_to_linear")
        
        # Mutação 3: Perturbação direcionada de pesos (30% chance)
        if random.random() < 0.30:
            with torch.no_grad():
                # Perturbação pequena e direcionada
                noise_scale = 0.01 * (1.0 + self.successful_mutations * 0.1)  # Aumenta com sucesso
                noise = torch.randn_like(self.fc.weight) * noise_scale
                self.fc.weight.data += noise
                mutations_applied.append(f"weight_perturbation_scale_{noise_scale:.4f}")
        
        # Mutação 4: Ajuste de bias (10% chance)
        if random.random() < 0.10:
            with torch.no_grad():
                bias_adjustment = (random.random() - 0.5) * 0.05
                self.fc.bias.data += bias_adjustment
                mutations_applied.append(f"bias_adjustment_{bias_adjustment:.4f}")
        
        self.mutation_count += len(mutations_applied)
        
        return await {
            "mutations_applied": mutations_applied,
            "total_mutations": len(mutations_applied),
            "cumulative_mutations": self.mutation_count
        }

    async def measure_novelty(self) -> float:
        """
        Mede novidade estrutural/paramétrica vs snapshot anterior
        """
        with torch.no_grad():
            # Distância L2 dos pesos
            weight_distance = torch.norm(
                self.fc.weight - self.last_weight_snapshot, p=2
            ).item()
            
            # Normalizar pela magnitude dos pesos
            weight_magnitude = torch.norm(self.fc.weight, p=2).item()
            normalized_distance = weight_distance / (weight_magnitude + 1e-8)
            
            return await normalized_distance

    async def update_weight_snapshot(self):
        """Atualiza snapshot dos pesos para próxima medição de novidade"""
        self.last_weight_snapshot.copy_(self.fc.weight.detach())

    async def calculate_consciousness(self, performance_metrics: Dict[str, Any]) -> float:
        """
        Calcula score de consciência baseado em:
        - Complexidade arquitetural
        - Capacidade adaptativa
        - Memória temporal (idade/experiência)
        - Auto-evolução bem-sucedida
        """
        # Complexidade arquitetural
        has_learnable_gate = isinstance(self.gate, nn.Linear)
        arch_complexity = 0.5 + (0.3 if has_learnable_gate else 0.0)
        
        # Capacidade adaptativa
        adaptation = min(1.0, abs(performance_metrics.get("delta_val_loss", 0.0)) * 50)
        
        # Memória temporal
        temporal_memory = min(1.0, self.age * 0.02)
        
        # Auto-evolução
        evolution_success = min(1.0, self.successful_mutations * 0.1)
        
        # Diversidade de ativação (entropia proxy)
        activation_diversity = 0.5  # Base para ativação fixa, pode aumentar com mixed activation
        
        # Fórmula composta
        consciousness = (
            0.2 * arch_complexity +
            0.25 * adaptation +
            0.15 * temporal_memory +
            0.25 * evolution_success +
            0.15 * activation_diversity
        )
        
        self.consciousness_score = max(0.0, min(1.0, consciousness))
        return await self.consciousness_score

    # ═══════════════════════════════════════════════════════════════════════════
    # TREINAMENTO E AVALIAÇÃO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def fit_eval(self, workdir: str) -> Dict[str, Any]:
        """
        Executa ciclo completo:
        1. Treino base
        2. Auto-mutação
        3. Treino pós-mutação
        4. Avaliação OOD
        5. Cálculo de métricas IA³
        """
        logger.info(f"\n🧬 Iniciando ciclo IA³ para neurônio {self.neuron_id}")
        logger.info(f"   Idade: {self.age}, Mutações: {self.mutation_count}")
        
        os.makedirs(workdir, exist_ok=True)
        self.age += 1
        
        # Gerar dados de treino e validação
        (X_train, y_train), (X_val, y_val) = _make_data(
            n=512, 
            d_in=self.cfg.input_dim, 
            seed=self.cfg.seed + self.age,
            complexity_level=2
        )
        
        # Dados OOD para teste de adaptação
        X_ood, y_ood = _make_ood_data(X_val, y_val, shift_factor=0.2)
        
        self.to(self.cfg.device)
        loss_fn = nn.MSELoss()
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 1: TREINO BASE
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(f"   📚 Treino base ({self.cfg.steps} steps)...")
        
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        
        self.train()
        train_losses = []
        
        for step in range(self.cfg.steps):
            # Batch aleatório
            indices = torch.randint(0, X_train.size(0), (self.cfg.batch,))
            x_batch, y_batch = X_train[indices], y_train[indices]
            
            # Forward e backward
            optimizer.zero_grad()
            pred = self.forward(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validação pré-mutação
        self.eval()
        with torch.no_grad():
            val_pred_pre = self.forward(X_val)
            base_val_loss = loss_fn(val_pred_pre, y_val).item()
            
            # OOD baseline
            ood_pred_pre = self.forward(X_ood)
            base_ood_loss = loss_fn(ood_pred_pre, y_ood).item()
        
        logger.info(f"      ✅ Loss treino: {train_losses[-1]:.6f}")
        logger.info(f"      ✅ Loss validação: {base_val_loss:.6f}")
        logger.info(f"      ✅ Loss OOD: {base_ood_loss:.6f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 2: AUTO-MUTAÇÃO
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(f"   🔬 Executando auto-mutação...")
        
        # Backup do estado atual
        state_backup = self.state_dict()
        config_backup = {
            "act": self.cfg.act,
            "gate_type": type(self.gate).__name__
        }
        
        # Propor mutações
        mutation_result = self.propose_self_mutation()
        mutations_applied = mutation_result["mutations_applied"]
        
        if mutations_applied:
            logger.info(f"      🔄 Mutações aplicadas: {mutations_applied}")
            
            # Treino pós-mutação (mais curto)
            post_mutation_optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr * 0.5)
            
            self.train()
            for step in range(self.cfg.steps // 2):  # Metade dos steps
                indices = torch.randint(0, X_train.size(0), (self.cfg.batch,))
                x_batch, y_batch = X_train[indices], y_train[indices]
                
                post_mutation_optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                post_mutation_optimizer.step()
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 3: AVALIAÇÃO PÓS-MUTAÇÃO
        # ═══════════════════════════════════════════════════════════════════════
        self.eval()
        with torch.no_grad():
            # Validação pós-mutação
            val_pred_post = self.forward(X_val)
            post_val_loss = loss_fn(val_pred_post, y_val).item()
            
            # OOD pós-mutação
            ood_pred_post = self.forward(X_ood)
            post_ood_loss = loss_fn(ood_pred_post, y_ood).item()
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 4: ANÁLISE DE SUCESSO E NOVIDADE
        # ═══════════════════════════════════════════════════════════════════════
        delta_val_loss = post_val_loss - base_val_loss
        delta_ood_loss = post_ood_loss - base_ood_loss
        novelty = self.measure_novelty()
        
        # Determinar sucesso da evolução
        self_evolve_success = (
            delta_val_loss < 0 and  # Melhorou na validação
            delta_ood_loss < 0.05   # Não piorou muito em OOD
        )
        
        if self_evolve_success:
            self.successful_mutations += 1
            self.update_weight_snapshot()  # Aceitar mudanças
            logger.info(f"      ✅ Auto-evolução BEM-SUCEDIDA!")
        else:
            # Rollback das mutações se não melhoraram
            self.load_state_dict(state_backup)
            self.act = ACTIVATIONS[config_backup["act"]]()
            if config_backup["gate_type"] == "Identity":
                self.gate = nn.Identity()
            else:
                self.gate = nn.Linear(1, 1, bias=False)
            logger.info(f"      🔄 Auto-evolução falhou, rollback aplicado")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FASE 5: MÉTRICAS FINAIS IA³
        # ═══════════════════════════════════════════════════════════════════════
        
        # Calcular consciência
        performance_metrics = {
            "delta_val_loss": delta_val_loss,
            "delta_ood_loss": delta_ood_loss,
            "novelty": novelty
        }
        consciousness = self.calculate_consciousness(performance_metrics)
        
        # Métricas completas
        final_metrics = {
            # Identificação
            "neuron_id": self.neuron_id,
            "age": self.age,
            "generation": self.generation,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            
            # Performance
            "base_val_loss": base_val_loss,
            "post_val_loss": post_val_loss,
            "delta_val_loss": delta_val_loss,
            "base_ood_loss": base_ood_loss,
            "post_ood_loss": post_ood_loss,
            "delta_ood_loss": delta_ood_loss,
            "best_val": post_val_loss,  # Para compatibilidade
            
            # Evolução
            "novelty": novelty,
            "self_evolve_success": self_evolve_success,
            "mutations_applied": mutations_applied,
            "self_mutations": len(mutations_applied),
            "total_mutations": self.mutation_count,
            "successful_mutations": self.successful_mutations,
            
            # IA³ específicos
            "consciousness_score": consciousness,
            "contribution_score": abs(self.get_gate_value()),
            "architectural_changes": len(mutations_applied),
            "synaptic_changes": self.mutation_count,
            "construction_mutations": len([m for m in mutations_applied if "gate" in m]),
            "autonomous_operation": True,  # Operou sem intervenção
            
            # Estado arquitetural
            "current_activation": self.cfg.act,
            "has_learnable_gate": isinstance(self.gate, nn.Linear),
            "weight_norm": self.get_weight_norm(),
            "grad_norm": self.get_grad_norm(),
            "architectural_complexity": self.get_architectural_complexity()
        }
        
        # Salvar relatório
        with open(os.path.join(workdir, "metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Salvar estado do neurônio
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.cfg,
            "metadata": {
                "neuron_id": self.neuron_id,
                "age": self.age,
                "mutation_count": self.mutation_count,
                "consciousness_score": consciousness
            }
        }, os.path.join(workdir, "neuron_state.pt"))
        
        logger.info(f"   📊 RELATÓRIO FINAL:")
        logger.info(f"      Δ Val Loss: {delta_val_loss:.6f}")
        logger.info(f"      Novidade: {novelty:.6f}")
        logger.info(f"      Sucesso evolução: {self_evolve_success}")
        logger.info(f"      Consciência: {consciousness:.3f}")
        logger.info(f"      Mutações: {len(mutations_applied)}")
        
        return await final_metrics

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTODOS AUXILIARES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_weight_norm(self) -> float:
        """Norma dos pesos do neurônio"""
        return await float(torch.norm(self.fc.weight, p=2).item())
    
    async def get_grad_norm(self) -> float:
        """Norma dos gradientes do neurônio"""
        if self.fc.weight.grad is None:
            return await 0.0
        return await float(torch.norm(self.fc.weight.grad, p=2).item())
    
    async def get_gate_value(self) -> float:
        """Valor efetivo do gate"""
        if isinstance(self.gate, nn.Linear):
            return await float(self.gate.weight.item())
        else:
            return await 1.0  # Identity gate
    
    async def get_architectural_complexity(self) -> float:
        """Score de complexidade arquitetural"""
        complexity = 0.5  # Base
        
        # Bonus por gate learnable
        if isinstance(self.gate, nn.Linear):
            complexity += 0.3
        
        # Bonus por ativação não-trivial
        if self.cfg.act != "relu":
            complexity += 0.2
        
        return await min(1.0, complexity)

    async def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo completo do neurônio"""
        return await {
            "id": self.neuron_id,
            "age": self.age,
            "generation": self.generation,
            "activation": self.cfg.act,
            "has_learnable_gate": isinstance(self.gate, nn.Linear),
            "consciousness": self.consciousness_score,
            "total_mutations": self.mutation_count,
            "successful_mutations": self.successful_mutations,
            "weight_norm": self.get_weight_norm(),
            "architectural_complexity": self.get_architectural_complexity(),
            "birth_time": self.birth_time.isoformat()
        }

if __name__ == "__main__":
    import sys
    
    # Teste da célula-neurônio
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 Testando célula-neurônio IA³...")
        
        cfg = TrainConfig(
            input_dim=16,
            hidden_dim=16,
            lr=0.01,
            steps=100,
            seed=42,
            batch=32
        )
        
        neuron = IA3Neuron(cfg)
        
        # Executar ciclo completo
        workdir = f"/tmp/neuron_test_{int(time.time())}"
        metrics = neuron.fit_eval(workdir)
        
        logger.info(f"\n📊 Resultados do teste:")
        logger.info(f"   Neurônio ID: {metrics['neuron_id']}")
        logger.info(f"   Δ Val Loss: {metrics['delta_val_loss']:.6f}")
        logger.info(f"   Novidade: {metrics['novelty']:.6f}")
        logger.info(f"   Sucesso evolução: {metrics['self_evolve_success']}")
        logger.info(f"   Consciência: {metrics['consciousness_score']:.3f}")
        logger.info(f"   Mutações: {metrics['self_mutations']}")
        
        # Testar Equação da Morte
        sys.path.append("/root/darwin")
        from rules.equacao_da_morte import equacao_da_morte
        
        death_result = equacao_da_morte(metrics)
        logger.info(f"\n⚖️ Equação da Morte:")
        logger.info(f"   Decisão: {death_result['decision']}")
        logger.info(f"   E(t+1): {death_result['E']}")
        logger.info(f"   A(t): {death_result['A']}, C(t): {death_result['C']}")
        
        logger.info(f"\n✅ Teste completo!")
        logger.info(f"   Workdir: {workdir}")
    else:
        logger.info("Uso: python ia3_neuron.py --test")