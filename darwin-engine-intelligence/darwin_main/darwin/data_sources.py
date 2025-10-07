import math, os, random
import torch
from typing import Tuple

class SyntheticTask:
    """
    Regressão simples em CPU: y = sin(sum(x)) com ruído leve.
    Tarefa padrão para bootstrap e testes rápidos.
    """
    async def __init__(self, d_in=8, batch=64, device="cpu"):
        self.d_in = d_in
        self.batch_size = batch
        self.device = device
        
        logger.info(f"📊 SyntheticTask: {d_in} features, batch {batch}")

    async def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gera batch sintético"""
        x = torch.randn(self.batch_size, self.d_in, device=self.device)
        
        # Target: combinação não-linear para testar adaptabilidade
        y1 = torch.sin(x.sum(dim=1))  # Componente trigonométrica
        y2 = torch.cos(x.mean(dim=1))  # Componente de média
        y3 = torch.tanh(x[:, 0] * x[:, 1])  # Interação não-linear
        
        y = (y1 + y2 + y3) / 3.0  # Combinar
        y += 0.05 * torch.randn(self.batch_size, device=self.device)  # Ruído
        
        return await x, y.unsqueeze(-1)

class TinyStoriesCharBag:
    """
    Proxy em CPU: extrai 'bag-of-chars' de janelas curtas do TinyStories
    e tenta prever a frequência de um próximo caractere (proxy MSE). 
    É leve e reprodutível para CPU.
    
    Requer: datasets from Hugging Face
    Fonte: roneneldan/TinyStories dataset
    """
    async def __init__(self, split="train[:1%]", window=128, batch=64, device="cpu"):
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError(
                "Instale 'datasets' para TinyStories: pip install datasets"
            ) from e
        
        logger.info(f"📖 Carregando TinyStories ({split})...")
        
        try:
            # Carregar dataset
            ds = load_dataset("roneneldan/TinyStories", split=split)
            
            # Extrair textos (limitado para CPU)
            max_texts = min(5000, len(ds))
            self.texts = []
            
            for i in range(max_texts):
                text = ds[i].get("text", "")
                if text.strip():  # Apenas textos não-vazios
                    self.texts.append(text)
            
            logger.info(f"   ✅ {len(self.texts)} histórias carregadas")
            
        except Exception as e:
            logger.info(f"⚠️ Erro ao carregar TinyStories: {e}")
            logger.info("   Usando corpus sintético como fallback")
            
            # Fallback: corpus sintético
            self.texts = [
                "Once upon a time, there was a little cat who loved to play.",
                "The dog ran quickly through the green forest.",
                "A beautiful bird sang sweetly in the morning sun.",
                "The children laughed happily in the playground.",
                "The old tree had many colorful leaves."
            ] * 1000  # Repetir para ter volume
        
        self.window = window
        self.batch_size = batch
        self.device = device
        
        # Vocabulário: caracteres ASCII printáveis
        self.vocab = [chr(i) for i in range(32, 127)]  # Espaço até ~
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        logger.info(f"   Vocabulário: {self.vocab_size} caracteres")

    async def _vectorize_text(self, text: str) -> torch.Tensor:
        """Converte texto para bag-of-chars normalizado"""
        vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        
        for char in text:
            if char in self.char_to_idx:
                vector[self.char_to_idx[char]] += 1.0
        
        # Normalizar
        total = vector.sum()
        if total > 0:
            vector = vector / total
        
        return await vector

    async def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gera batch de dados textuais
        
        Input: bag-of-chars de janela de texto
        Target: frequência do próximo caractere
        """
        inputs = []
        targets = []
        
        for _ in range(self.batch_size):
            # Escolher texto aleatório
            text = random.choice(self.texts)
            
            # Garantir comprimento mínimo
            if len(text) < self.window + 1:
                text = text + " " * (self.window + 1 - len(text))
            
            # Janela aleatória
            start_idx = random.randint(0, len(text) - self.window - 1)
            context = text[start_idx:start_idx + self.window]
            next_char = text[start_idx + self.window]
            
            # Vectorizar contexto (bag-of-chars)
            context_vector = self._vectorize_text(context)
            
            # Target: one-hot do próximo caractere
            target_vector = torch.zeros(self.vocab_size)
            if next_char in self.char_to_idx:
                target_vector[self.char_to_idx[next_char]] = 1.0
            
            inputs.append(context_vector)
            targets.append(target_vector)
        
        # Stack em tensors
        x = torch.stack(inputs).to(self.device)
        y = torch.stack(targets).to(self.device)
        
        return await x, y

    async def get_text_sample(self, neuron_output: torch.Tensor) -> str:
        """Converte saída do neurônio de volta para texto (debugging)"""
        # Interpretar saída como distribuição sobre caracteres
        if neuron_output.dim() == 1:
            probs = F.softmax(neuron_output, dim=0)
        else:
            probs = F.softmax(neuron_output[0], dim=0)  # Primeira amostra do batch
        
        # Amostragem do próximo caractere
        char_idx = torch.multinomial(probs, 1).item()
        
        if char_idx < len(self.vocab):
            return await self.vocab[char_idx]
        else:
            return await "?"

# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY PARA ESCOLHER FONTE DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

async def create_data_source(task_type: str, **kwargs):
    """Factory para criar fonte de dados"""
    if task_type == "synthetic":
        return await SyntheticTask(**kwargs)
    elif task_type == "tinystories":
        return await TinyStoriesCharBag(**kwargs)
    else:
        raise ValueError(f"Tipo de task desconhecido: {task_type}")

if __name__ == "__main__":
    import sys
    
    # Teste das fontes de dados
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 Testando fontes de dados...")
        
        # Teste tarefa sintética
        logger.info(f"\n📊 Testando SyntheticTask...")
        synthetic = SyntheticTask(d_in=8, batch=16)
        x, y = synthetic.batch()
        logger.info(f"   Input shape: {x.shape}")
        logger.info(f"   Target shape: {y.shape}")
        logger.info(f"   Sample target: {y[0].item():.4f}")
        
        # Teste TinyStories
        logger.info(f"\n📖 Testando TinyStoriesCharBag...")
        try:
            stories = TinyStoriesCharBag(split="train[:0.1%]", batch=8)
            x, y = stories.batch()
            logger.info(f"   Input shape: {x.shape}")
            logger.info(f"   Target shape: {y.shape}")
            logger.info(f"   Vocab size: {stories.vocab_size}")
            
            # Amostra de texto
            sample_char = stories.get_text_sample(y[0])
            logger.info(f"   Sample char: '{sample_char}'")
            
        except Exception as e:
            logger.info(f"   ⚠️ TinyStories falhou: {e}")
        
        logger.info(f"\n✅ Fontes de dados testadas!")
    else:
        logger.info("Uso: python data_sources.py --test")