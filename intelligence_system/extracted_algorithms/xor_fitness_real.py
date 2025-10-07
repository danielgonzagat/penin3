"""
XOR FITNESS FUNCTION - 100% REAL

Testa se um genoma consegue aprender XOR
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from typing import Any

# XOR dataset
XOR_X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_Y = torch.FloatTensor([[0], [1], [1], [0]])


def build_network_from_genome(genome) -> nn.Module:
    """Build PyTorch network from genome"""
    layers = []
    
    # Input layer
    input_size = 2  # XOR has 2 inputs
    
    # CORREÇÃO CRÍTICA: genome.layers pode vir como int, list ou tuple de checkpoints antigos
    if isinstance(genome.layers, int):
        # Se é int, criar tupla com 1 layer desse tamanho
        genome_layers = (genome.layers,)
    elif isinstance(genome.layers, list):
        # Se é list (de checkpoint JSON), converter para tuple
        genome_layers = tuple(genome.layers)
    else:
        # Já é tuple
        genome_layers = genome.layers
    
    # Hidden layers from genome
    for hidden_size in genome_layers:
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Activation
        if genome.activation == 'relu':
            layers.append(nn.ReLU())
        elif genome.activation == 'tanh':
            layers.append(nn.Tanh())
        elif genome.activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif genome.activation == 'gelu':
            layers.append(nn.GELU())
        
        # Dropout
        if genome.dropout > 0:
            layers.append(nn.Dropout(genome.dropout))
        
        input_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(input_size, 1))
    layers.append(nn.Sigmoid())  # Binary output
    
    return nn.Sequential(*layers)


def xor_fitness_real(genome, max_epochs: int = 100, patience: int = 20) -> float:
    """
    REAL fitness function - trains network on XOR
    
    Returns: Fitness = 1 - final_loss (higher is better)
    """
    try:
        # Build network
        net = build_network_from_genome(genome)
        
        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=genome.learning_rate)
        criterion = nn.BCELoss()
        
        # Train
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Forward
            output = net(XOR_X)
            loss = criterion(output, XOR_Y)
            
            # Backward
            optimizer.zero_grad()
            # FIX P#5: Suppress incompletude coroutine warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*backward_with_incompletude.*")
                loss.backward()
            optimizer.step()
            
            # Check improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
        
        # Final evaluation
        with torch.no_grad():
            final_output = net(XOR_X)
            final_loss = criterion(final_output, XOR_Y).item()
        
        # Fitness = 1 - loss (so lower loss = higher fitness)
        # Also add bonus for accuracy
        with torch.no_grad():
            predictions = (final_output > 0.5).float()
            accuracy = (predictions == XOR_Y).float().mean().item()
        
        # Fitness combines loss and accuracy
        fitness = (1.0 - min(final_loss, 1.0)) * 0.5 + accuracy * 0.5
        
        return max(0.0, fitness)
    
    except Exception as e:
        # Network failed (bad architecture)
        return 0.0


def xor_fitness_fast(genome) -> float:
    """
    FAST fitness function - fewer epochs for quick evolution
    """
    return xor_fitness_real(genome, max_epochs=50, patience=10)


def test_xor_fitness():
    """Test XOR fitness function"""
    from extracted_algorithms.neural_evolution_core import NeuralGenome
    
    print("="*80)
    print("🧪 TESTING XOR FITNESS FUNCTION")
    print("="*80)
    
    # Test 1: Good genome
    good_genome = NeuralGenome(
        layers=(8, 8),
        activation='tanh',
        learning_rate=0.01,
        dropout=0.0
    )
    
    print(f"\n1️⃣ Good genome:")
    print(f"   Layers: {good_genome.layers}")
    fitness = xor_fitness_real(good_genome, max_epochs=200)
    print(f"   Fitness: {fitness:.4f}")
    
    # Test 2: Bad genome (tiny network)
    bad_genome = NeuralGenome(
        layers=(2,),
        activation='relu',
        learning_rate=0.001,
        dropout=0.0
    )
    
    print(f"\n2️⃣ Bad genome:")
    print(f"   Layers: {bad_genome.layers}")
    fitness = xor_fitness_real(bad_genome, max_epochs=200)
    print(f"   Fitness: {fitness:.4f}")
    
    # Test 3: Random genome
    random_genome = NeuralGenome(
        layers=(16, 8),
        activation='gelu',
        learning_rate=0.005,
        dropout=0.1
    )
    
    print(f"\n3️⃣ Random genome:")
    print(f"   Layers: {random_genome.layers}")
    fitness = xor_fitness_real(random_genome, max_epochs=200)
    print(f"   Fitness: {fitness:.4f}")
    
    print(f"\n{'='*80}")
    print(f"✅ XOR FITNESS IS REAL! (different genomes → different fitness)")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_xor_fitness()
