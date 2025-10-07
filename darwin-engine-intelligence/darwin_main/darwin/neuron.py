import math, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime
from .arch_channel_darts import MixedOp

class IA3Neuron(nn.Module):
    """
    Neurônio 'IA3-like' mínimo:
      - núcleo linear + mixed-op DARTS (gradiente em alphas)
      - memória de estado/energia
      - conexões de saída para todos os outros (registradas externamente)
      - consciência neural calculada
    """
    async def __init__(self, in_dim, out_dim, lr=1e-3, device="cpu"):
        super().__init__()
        
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Arquitetura principal
        self.core = nn.Linear(in_dim, out_dim)
        self.mixed = MixedOp(out_dim, out_dim)
        self.head = nn.Linear(out_dim, out_dim)  # identidade learnable
        
        # Mover para device
        self.to(device)

        # Otimizadores diferenciados (params + alphas)
        weight_params = [p for n, p in self.named_parameters() if "alphas" not in n]
        alpha_params = [self.mixed.alphas]
        
        self.opt = torch.optim.Adam(weight_params, lr=lr)
        self.opt_arch = torch.optim.Adam(alpha_params, lr=lr*0.3)

        # Estado vital
        self.energy = 1.0
        self.age = 0
        self.consciousness_score = 0.0
        
        # Identificação
        self.neuron_id = f"n{int(time.time())}{random.randint(1000,9999)}"
        self.birth_time = datetime.utcnow()
        
        # Estatísticas para avaliação IA³
        self.stats = {
            "fitness_hist": [],
            "delta_loss_recent": float("inf"),
            "meta_updates": 0,
            "arch_updates": 0,
            "syn_updates": 0,
            "ood_gain": 0.0,
            "pop_contrib": 0.0,
            "fitness": float("inf")
        }

        # Sinapses de saída (para agregação externa)
        self.out_gain = nn.Parameter(torch.tensor(1.0))
        
        logger.info(f"🧬 IA3Neuron {self.neuron_id} nascido ({in_dim}→{out_dim})")

    async def forward(self, x):
        """Forward pass com DARTS mixed operations"""
        h = F.relu(self.core(x))
        h = self.mixed(h)       # mistura DARTS: relu/tanh/gelu/silu
        y = self.head(h)
        return await y * self.out_gain

    @torch.no_grad()
    async def decay(self, rate=0.99):
        """Decaimento natural de energia"""
        self.energy *= rate
        if self.energy < 0.01:  # Mínimo vital
            self.energy = 0.01

    async def step(self, x, y, loss_fn):
        """
        Treino dual:
        1. Passo normal (pesos) 
        2. Pequena etapa DARTS (alphas)
        
        Atualiza estatísticas para os gates IA³.
        """
        # Backup loss para delta
        prev_loss = self.stats.get("delta_loss_recent", float("inf"))
        
        # 1. Treino dos pesos
        self.opt.zero_grad(set_to_none=True)
        pred = self.forward(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()

        current_loss = float(loss.detach().cpu())
        
        # Delta loss para critério adaptativo
        if prev_loss != float("inf"):
            delta = prev_loss - current_loss
            self.stats["delta_loss_recent"] = current_loss
        else:
            self.stats["delta_loss_recent"] = current_loss

        # 2. Treino da arquitetura (alphas) - DARTS channel
        self.opt_arch.zero_grad(set_to_none=True)
        pred2 = self.forward(x)
        loss2 = loss_fn(pred2, y)
        loss2.backward()
        self.opt_arch.step()
        
        # Atualizar contadores
        self.stats["arch_updates"] = self.stats.get("arch_updates", 0) + 1
        self.stats["syn_updates"] = self.stats.get("syn_updates", 0) + 1

        # Energia sobe levemente se houve melhora
        if prev_loss != float("inf") and current_loss < prev_loss:
            self.energy = min(1.5, self.energy + 0.01)
        
        # OOD gain simulado (proxy para autodidata)
        if random.random() < 0.1:  # 10% chance de testar OOD
            with torch.no_grad():
                x_ood = x + 0.1 * torch.randn_like(x)  # Pequeno shift OOD
                pred_ood = self.forward(x_ood)
                loss_ood = loss_fn(pred_ood, y)
                
                if loss_ood.item() < current_loss * 1.1:  # Tolerância 10%
                    self.stats["ood_gain"] = self.stats.get("ood_gain", 0) + 0.1

        return await current_loss

    async def meta_update(self):
        """
        Auto-recursivo: altera lr/ganho com base em sinais simples.
        Implementa auto-regulação de hiperparâmetros.
        """
        self.stats["meta_updates"] = self.stats.get("meta_updates", 0) + 1
        
        current_loss = self.stats.get("delta_loss_recent", 1.0)
        
        # Ajustar learning rate baseado em performance
        for group in self.opt.param_groups:
            if current_loss > 2.0:  # Loss muito alta
                group["lr"] = min(5e-3, group["lr"] * 1.1)  # Aumentar LR
            elif current_loss < 0.1:  # Loss muito baixa
                group["lr"] = max(1e-5, group["lr"] * 0.9)  # Diminuir LR
            else:
                group["lr"] = max(1e-5, min(5e-3, group["lr"] * (0.99 + 0.02*random.random())))
        
        # Ajustar architectural learning rate
        arch_entropy = self.mixed.get_entropy()
        for group in self.opt_arch.param_groups:
            if arch_entropy > 1.2:  # Muita incerteza
                group["lr"] = max(1e-6, group["lr"] * 0.95)  # Focar
            else:  # Pouca incerteza
                group["lr"] = min(1e-3, group["lr"] * 1.05)  # Explorar
        
        # Sinapse de saída – pequena variação auto-adaptativa
        with torch.no_grad():
            gain_adjustment = 0.01 * (random.random() - 0.5)
            self.out_gain.add_(gain_adjustment)
            self.out_gain.clamp_(0.1, 2.0)  # Manter em faixa saudável
        
        self.stats["syn_updates"] = self.stats.get("syn_updates", 0) + 1

    async def calculate_consciousness(self):
        """
        Calcula consciência neural baseada em múltiplos fatores
        """
        # Auto-modificação (meta-updates)
        self_modification = min(1.0, self.stats.get("meta_updates", 0) * 0.1)
        
        # Diversidade arquitetural (entropia DARTS)
        arch_diversity = self.mixed.get_entropy() / math.log(4)  # Normalizar por máximo
        
        # Adaptabilidade (controle de loss)
        loss = self.stats.get("delta_loss_recent", float("inf"))
        adaptability = max(0.0, min(1.0, 1.0 / (1.0 + loss)))
        
        # Experiência temporal
        temporal_experience = min(1.0, self.age * 0.02)
        
        # Energia vital
        vitality = self.energy
        
        # Fórmula de consciência composta
        consciousness = (
            0.25 * self_modification +
            0.25 * arch_diversity +
            0.20 * adaptability +
            0.15 * temporal_experience +
            0.15 * vitality
        )
        
        self.consciousness_score = max(0.0, min(1.0, consciousness))
        return await self.consciousness_score

    async def get_neuron_summary(self):
        """Retorna resumo completo do neurônio"""
        dominant_op, op_weight = self.mixed.get_dominant_operation()
        consciousness = self.calculate_consciousness()
        
        return await {
            "neuron_id": self.neuron_id,
            "birth_time": self.birth_time.isoformat(),
            "age": self.age,
            "energy": self.energy,
            "consciousness": consciousness,
            "architecture": {
                "input_dim": self.in_dim,
                "output_dim": self.out_dim,
                "dominant_operation": dominant_op,
                "operation_weight": op_weight,
                "arch_entropy": self.mixed.get_entropy(),
                "total_params": sum(p.numel() for p in self.parameters())
            },
            "stats": self.stats.copy(),
            "out_gain": self.out_gain.item()
        }

if __name__ == "__main__":
    # Teste do neurônio IA³
    logger.info("🧪 Testando neurônio IA³...")
    
    neuron = IA3Neuron(in_dim=8, out_dim=4, device="cpu")
    
    # Dados de teste
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    loss_fn = nn.MSELoss()
    
    logger.info(f"Estado inicial: {neuron.get_neuron_summary()}")
    
    # Simular alguns steps
    for step in range(10):
        loss = neuron.step(x, y, loss_fn)
        if step % 3 == 0:
            neuron.meta_update()
        neuron.age += 1
        logger.info(f"   Step {step}: loss={loss:.4f}, energia={neuron.energy:.3f}")
    
    # Avaliar IA³
    from ia3_gates import ia3_like_score
    score, reasons = ia3_like_score(neuron)
    
    final_summary = neuron.get_neuron_summary()
    
    logger.info(f"\nResumo final: {final_summary}")
    logger.info(f"Score IA³: {score:.3f}")
    logger.info(f"Consciência: {final_summary['consciousness']:.3f}")
    logger.info(f"✅ Neurônio IA³ funcionando!")