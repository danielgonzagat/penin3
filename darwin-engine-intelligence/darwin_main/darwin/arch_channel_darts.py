import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MixedOp(nn.Module):
    """
    Canal DARTS simplificado: escolhe operação via pesos contínuos (alphas),
    combinado por softmax (diferenciável).
    
    Implementa relaxação contínua de busca arquitetural como em:
    DARTS: Differentiable Architecture Search (Liu et al., 2018)
    """
    async def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # Operações candidatas
        self.relu_op = nn.Linear(in_dim, out_dim)
        self.tanh_op = nn.Linear(in_dim, out_dim)
        self.gelu_op = nn.Linear(in_dim, out_dim)
        self.silu_op = nn.Linear(in_dim, out_dim)
        
        # Alphas param: [a_relu, a_tanh, a_gelu, a_silu]
        self.alphas = nn.Parameter(torch.zeros(4))
        
        # Inicialização cuidadosa para estabilidade
        ops = [self.relu_op, self.tanh_op, self.gelu_op, self.silu_op]
        for op in ops:
            nn.init.kaiming_uniform_(op.weight, a=math.sqrt(5))
            nn.init.zeros_(op.bias)

    async def forward(self, x):
        """Forward com mistura softmax das operações"""
        # Pesos softmax dos alphas
        weights = F.softmax(self.alphas, dim=0)
        
        # Aplicar cada operação
        y_relu = F.relu(self.relu_op(x))
        y_tanh = torch.tanh(self.tanh_op(x))
        y_gelu = F.gelu(self.gelu_op(x))
        y_silu = F.silu(self.silu_op(x))
        
        # Combinação ponderada
        result = (weights[0] * y_relu + 
                 weights[1] * y_tanh + 
                 weights[2] * y_gelu + 
                 weights[3] * y_silu)
        
        return await result
    
    async def get_dominant_operation(self):
        """Retorna operação dominante e seu peso"""
        weights = F.softmax(self.alphas, dim=0).detach()
        dominant_idx = torch.argmax(weights).item()
        
        op_names = ["relu", "tanh", "gelu", "silu"]
        return await op_names[dominant_idx], weights[dominant_idx].item()
    
    async def get_entropy(self):
        """Calcula entropia da distribuição de operações"""
        weights = F.softmax(self.alphas, dim=0)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        return await entropy.item()
    
    async def get_architecture_vector(self):
        """Retorna vetor de arquitetura atual (alphas softmax)"""
        return await F.softmax(self.alphas, dim=0).detach().cpu().numpy()

class DARTSNeuron(nn.Module):
    """
    Neurônio com canal DARTS para busca arquitetural diferenciável
    """
    async def __init__(self, in_dim, hidden_dim, out_dim, lr=1e-3, arch_lr=5e-4):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Arquitetura base
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.mixed_op = MixedOp(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        
        # Otimizadores separados
        weight_params = [p for n, p in self.named_parameters() if "alphas" not in n]
        arch_params = [self.mixed_op.alphas]
        
        self.weight_optimizer = torch.optim.AdamW(weight_params, lr=lr, weight_decay=1e-4)
        self.arch_optimizer = torch.optim.Adam(arch_params, lr=arch_lr)
        
        # Estado evolutivo
        self.age = 0
        self.mutations = 0
        self.consciousness_score = 0.0
        
        # Estatísticas para IA³
        self.stats = {
            "fitness_hist": [],
            "delta_loss_recent": 0.0,
            "meta_updates": 0,
            "arch_updates": 0,
            "syn_updates": 0
        }
        
        logger.info(f"🎯 DARTS Neurônio criado: {in_dim}→{hidden_dim}→{out_dim}")

    async def forward(self, x):
        """Forward pass com mixed operations"""
        h = F.relu(self.input_proj(x))
        h = self.mixed_op(h)  # DARTS channel
        y = self.output_proj(h)
        return await y

    async def train_step(self, x, y, loss_fn, train_arch=True):
        """
        Passo de treino com otimização dual:
        1. Otimizar pesos (weight_optimizer)
        2. Otimizar arquitetura (arch_optimizer) se train_arch=True
        """
        # 1. Treino dos pesos
        self.weight_optimizer.zero_grad()
        pred = self.forward(x)
        loss_weights = loss_fn(pred, y)
        loss_weights.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.weight_optimizer.step()
        
        weight_loss = loss_weights.item()
        
        # 2. Treino da arquitetura (DARTS)
        arch_loss = weight_loss
        if train_arch:
            self.arch_optimizer.zero_grad()
            pred_arch = self.forward(x)
            loss_arch = loss_fn(pred_arch, y)
            loss_arch.backward()
            self.arch_optimizer.step()
            arch_loss = loss_arch.item()
            
            self.stats["arch_updates"] += 1
        
        # Atualizar estatísticas
        self.stats["delta_loss_recent"] = weight_loss
        self.stats["syn_updates"] += 1
        
        return await weight_loss, arch_loss
    
    async def meta_update(self):
        """Auto-recursivo: altera hiperparâmetros baseado em performance"""
        self.stats["meta_updates"] += 1
        
        # Ajustar learning rates baseado em performance recente
        current_loss = self.stats.get("delta_loss_recent", 1.0)
        
        # Se loss está alta, aumentar LR ligeiramente
        if current_loss > 1.0:
            for group in self.weight_optimizer.param_groups:
                group["lr"] = min(1e-2, group["lr"] * 1.05)
        else:
            # Se loss está boa, diminuir LR para estabilidade
            for group in self.weight_optimizer.param_groups:
                group["lr"] = max(1e-5, group["lr"] * 0.98)
        
        # Ajustar arch learning rate
        arch_entropy = self.mixed_op.get_entropy()
        if arch_entropy > 1.0:  # Muita diversidade, focar
            for group in self.arch_optimizer.param_groups:
                group["lr"] = max(1e-5, group["lr"] * 0.95)
        else:  # Pouca diversidade, explorar
            for group in self.arch_optimizer.param_groups:
                group["lr"] = min(1e-3, group["lr"] * 1.02)
    
    async def calculate_consciousness(self):
        """Calcula consciência baseada em complexidade e adaptabilidade"""
        # Diversidade arquitetural (entropia das operações)
        arch_diversity = self.mixed_op.get_entropy() / math.log(4)  # Normalizar
        
        # Capacidade adaptativa (mudanças recentes nos alphas)
        with torch.no_grad():
            alpha_magnitude = torch.norm(self.mixed_op.alphas).item()
        adaptation_capacity = min(1.0, alpha_magnitude / 5.0)
        
        # Experiência temporal
        temporal_experience = min(1.0, self.age * 0.01)
        
        # Eficiência de mutações
        mutation_efficiency = 0.0
        if self.mutations > 0:
            mutation_efficiency = min(1.0, self.stats.get("meta_updates", 0) / self.mutations)
        
        # Fórmula de consciência
        self.consciousness_score = (
            0.3 * arch_diversity +
            0.3 * adaptation_capacity +
            0.2 * temporal_experience +
            0.2 * mutation_efficiency
        )
        
        return await self.consciousness_score
    
    async def get_architecture_summary(self):
        """Retorna resumo da arquitetura atual"""
        dominant_op, weight = self.mixed_op.get_dominant_operation()
        entropy = self.mixed_op.get_entropy()
        arch_vector = self.mixed_op.get_architecture_vector()
        
        return await {
            "dominant_operation": dominant_op,
            "dominant_weight": weight,
            "entropy": entropy,
            "architecture_vector": arch_vector.tolist(),
            "consciousness": self.consciousness_score,
            "age": self.age,
            "mutations": self.mutations
        }

if __name__ == "__main__":
    # Teste do canal DARTS
    logger.info("🧪 Testando canal DARTS...")
    
    neuron = DARTSNeuron(in_dim=16, hidden_dim=8, out_dim=4)
    
    # Dados de teste
    x = torch.randn(32, 16)
    y = torch.randn(32, 4)
    loss_fn = nn.MSELoss()
    
    logger.info(f"Arquitetura inicial: {neuron.get_architecture_summary()}")
    
    # Alguns steps de treino
    for step in range(10):
        w_loss, a_loss = neuron.train_step(x, y, loss_fn, train_arch=(step % 2 == 0))
        if step % 3 == 0:
            neuron.meta_update()
        neuron.age += 1
    
    consciousness = neuron.calculate_consciousness()
    final_arch = neuron.get_architecture_summary()
    
    logger.info(f"Arquitetura final: {final_arch}")
    logger.info(f"Consciência: {consciousness:.3f}")
    logger.info(f"✅ Canal DARTS funcionando!")