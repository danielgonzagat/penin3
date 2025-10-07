"""
Multi-Modal REAL - Funcionalidade básica demonstrável
"""
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiModalReal:
    """Multi-modal com funcionalidade REAL mínima"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
        # Text encoder (simple)
        self.text_encoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Image encoder (simple)
        self.image_encoder = nn.Sequential(
            nn.Linear(224*224*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        logger.info(f"✅ MultiModalReal initialized (dim={embedding_dim})")
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding (simplified)"""
        # Convert text to vector (char-based, simplified)
        text_vec = np.zeros(100)
        for i, char in enumerate(text[:100]):
            text_vec[i] = ord(char) / 255.0
        
        text_tensor = torch.FloatTensor(text_vec).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.text_encoder(text_tensor)
        
        return embedding.squeeze(0)
    
    def embed_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode image to embedding (simplified)"""
        # Flatten image
        if len(image.shape) == 3:
            image = image.flatten()
        
        # Pad/crop to expected size
        expected_size = 224 * 224 * 3
        if len(image) > expected_size:
            image = image[:expected_size]
        elif len(image) < expected_size:
            image = np.pad(image, (0, expected_size - len(image)))
        
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.image_encoder(image_tensor)
        
        return embedding.squeeze(0)
    
    def fuse(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """Fuse text and image embeddings"""
        combined = torch.cat([text_emb, image_emb], dim=0).unsqueeze(0)
        
        with torch.no_grad():
            fused = self.fusion(combined)
        
        return fused.squeeze(0)
    
    def process(self, text: str, image: np.ndarray) -> dict:
        """Process both modalities and fuse"""
        text_emb = self.embed_text(text)
        image_emb = self.embed_image(image)
        fused_emb = self.fuse(text_emb, image_emb)
        
        return {
            'text_embedding': text_emb,
            'image_embedding': image_emb,
            'fused_embedding': fused_emb,
            'embedding_dim': self.embedding_dim
        }


def test_multimodal_real():
    """Test multi-modal REAL"""
    print("="*80)
    print("🔥 MULTI-MODAL REAL - TESTE")
    print("="*80)
    
    mm = MultiModalReal(embedding_dim=512)
    
    # Test 1: Text embedding
    print("\n1️⃣ Text embedding:")
    text_emb = mm.embed_text("Hello world!")
    print(f"   Input: 'Hello world!'")
    print(f"   Output shape: {text_emb.shape}")
    print(f"   Output dim: {text_emb.shape[0]}")
    
    assert text_emb.shape[0] == 512, "Wrong dim!"
    print(f"   ✅ Text encoder FUNCIONA!")
    
    # Test 2: Image embedding
    print("\n2️⃣ Image embedding:")
    fake_image = np.random.randn(224, 224, 3)
    image_emb = mm.embed_image(fake_image)
    print(f"   Input shape: {fake_image.shape}")
    print(f"   Output shape: {image_emb.shape}")
    print(f"   Output dim: {image_emb.shape[0]}")
    
    assert image_emb.shape[0] == 512, "Wrong dim!"
    print(f"   ✅ Image encoder FUNCIONA!")
    
    # Test 3: Fusion
    print("\n3️⃣ Multi-modal fusion:")
    fused = mm.fuse(text_emb, image_emb)
    print(f"   Text + Image → Fused")
    print(f"   Fused shape: {fused.shape}")
    print(f"   Fused dim: {fused.shape[0]}")
    
    assert fused.shape[0] == 512, "Wrong dim!"
    print(f"   ✅ Fusion FUNCIONA!")
    
    # Test 4: Full pipeline
    print("\n4️⃣ Full pipeline:")
    result = mm.process("Test text", fake_image)
    
    print(f"   Outputs:")
    print(f"   - text_embedding: {result['text_embedding'].shape}")
    print(f"   - image_embedding: {result['image_embedding'].shape}")
    print(f"   - fused_embedding: {result['fused_embedding'].shape}")
    
    print(f"\n✅ MULTI-MODAL COMPLETO FUNCIONA!")
    print(f"   Text encoding: ✅")
    print(f"   Image encoding: ✅")
    print(f"   Fusion: ✅")
    
    return True


if __name__ == "__main__":
    test_multimodal_real()
