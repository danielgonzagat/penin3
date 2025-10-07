"""
Quantum-Inspired Optimization
Classical algorithms inspired by quantum principles (superposition, interference, entanglement)

NOTE: This is NOT real quantum computing - it's classical algorithms that mimic quantum behavior
"""

import numpy as np
import logging
from typing import Callable, List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class QuantumInspiredOptimizer:
    """
    Optimization using quantum-inspired principles
    
    Key concepts:
    1. Superposition: Represent multiple solutions simultaneously
    2. Interference: Amplify good solutions, suppress bad ones
    3. Measurement: Probabilistic sampling from superposition
    4. Entanglement: Correlate related parameters
    
    Based on:
    - Quantum Annealing
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Quantum-inspired Evolutionary Algorithm (QEA)
    """
    
    def __init__(self, n_qubits: int = 10, seed: int = 42):
        """
        Initialize quantum-inspired optimizer
        
        Args:
            n_qubits: Number of "qubits" (search space dimensionality)
            seed: Random seed
        """
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        
        # Amplitude (quantum state vector)
        # Initially in uniform superposition
        self.amplitude = np.ones(self.n_states, dtype=complex) / np.sqrt(self.n_states)
        
        # Best solution found
        self.best_state = None
        self.best_fitness = -np.inf
        
        # History
        self.history: List[Dict] = []
        
        np.random.seed(seed)
        
        logger.info(f"⚛️ QuantumInspiredOptimizer initialized: {n_qubits} qubits, {self.n_states} states")
    
    def measure(self) -> int:
        """
        'Measure' quantum state (collapse superposition)
        
        Returns:
            Measured state (integer 0 to 2^n_qubits - 1)
        """
        # Probabilities from amplitudes (Born rule)
        probabilities = np.abs(self.amplitude) ** 2
        probabilities /= probabilities.sum()  # Normalize
        
        # Sample
        measured_state = np.random.choice(self.n_states, p=probabilities)
        
        return measured_state
    
    def apply_interference(self, good_states: List[int], bad_states: List[int], 
                          amplification: float = 1.1, suppression: float = 0.9):
        """
        Apply quantum interference
        
        Args:
            good_states: States to amplify
            bad_states: States to suppress
            amplification: Factor to amplify (>1.0)
            suppression: Factor to suppress (<1.0)
        """
        # Amplify good states
        for state in good_states:
            if 0 <= state < self.n_states:
                self.amplitude[state] *= amplification
        
        # Suppress bad states
        for state in bad_states:
            if 0 <= state < self.n_states:
                self.amplitude[state] *= suppression
        
        # Renormalize
        norm = np.linalg.norm(self.amplitude)
        if norm > 0:
            self.amplitude /= norm
    
    def apply_rotation(self, theta: float):
        """
        Apply quantum rotation (phase shift)
        
        Args:
            theta: Rotation angle
        """
        # Apply phase rotation
        phase = np.exp(1j * theta)
        self.amplitude *= phase
    
    def optimize(self, 
                 fitness_fn: Callable[[int], float],
                 n_iterations: int = 100,
                 n_measurements: int = 10,
                 exploitation_rate: float = 0.7) -> Tuple[int, float]:
        """
        Quantum-inspired optimization
        
        Args:
            fitness_fn: Function to optimize (maps state to fitness)
            n_iterations: Number of optimization iterations
            n_measurements: Measurements per iteration
            exploitation_rate: Balance exploration vs exploitation
        
        Returns:
            (best_state, best_fitness)
        """
        logger.info(f"⚛️ Starting quantum-inspired optimization ({n_iterations} iterations)...")
        
        for iteration in range(n_iterations):
            # Measure multiple times (quantum parallelism)
            states = [self.measure() for _ in range(n_measurements)]
            
            # Evaluate fitness
            fitnesses = [fitness_fn(state) for state in states]
            
            # Track best
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_idx]
                self.best_state = states[max_idx]
                logger.info(f"   🏆 New best: state={self.best_state}, fitness={self.best_fitness:.4f}")
            
            # Separate good and bad states
            avg_fitness = np.mean(fitnesses)
            threshold = avg_fitness * exploitation_rate
            
            good_states = [s for s, f in zip(states, fitnesses) if f >= threshold]
            bad_states = [s for s, f in zip(states, fitnesses) if f < threshold]
            
            # Apply interference
            self.apply_interference(good_states, bad_states)
            
            # Apply quantum rotation (exploration)
            rotation_angle = np.pi / (10 + iteration / 10)  # Decrease with time
            self.apply_rotation(rotation_angle)
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'avg_fitness': avg_fitness,
                'best_fitness': self.best_fitness,
                'n_good_states': len(good_states)
            })
            
            if (iteration + 1) % 20 == 0:
                logger.info(f"   Iter {iteration + 1}: best={self.best_fitness:.4f}, avg={avg_fitness:.4f}")
        
        logger.info(f"   ✅ Optimization complete: best_fitness={self.best_fitness:.4f}")
        
        return self.best_state, self.best_fitness
    
    def decode_state(self, state: int) -> np.ndarray:
        """
        Decode integer state to parameter vector
        
        Args:
            state: Integer state (0 to 2^n_qubits - 1)
        
        Returns:
            Parameter vector
        """
        # Convert state to binary representation
        binary = format(state, f'0{self.n_qubits}b')
        
        # Each bit → continuous parameter in [0, 1]
        params = np.array([int(bit) for bit in binary], dtype=float)
        
        return params
    
    def optimize_continuous(self,
                           fitness_fn: Callable[[np.ndarray], float],
                           bounds: List[Tuple[float, float]],
                           n_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Optimize continuous parameters
        
        Args:
            fitness_fn: Function taking parameter vector
            bounds: [(min, max)] for each parameter
            n_iterations: Iterations
        
        Returns:
            (best_params, best_fitness)
        """
        n_params = len(bounds)
        
        # Wrapper function
        def discrete_fitness(state: int) -> float:
            # Decode to parameters
            params_normalized = self.decode_state(state)
            
            # Scale to bounds
            params = []
            for i, (min_val, max_val) in enumerate(bounds):
                if i < len(params_normalized):
                    param = min_val + (max_val - min_val) * params_normalized[i]
                else:
                    param = (min_val + max_val) / 2
                params.append(param)
            
            return fitness_fn(np.array(params))
        
        # Run discrete optimization
        best_state, best_fitness = self.optimize(discrete_fitness, n_iterations)
        
        # Decode best state
        params_normalized = self.decode_state(best_state)
        best_params = []
        for i, (min_val, max_val) in enumerate(bounds):
            if i < len(params_normalized):
                param = min_val + (max_val - min_val) * params_normalized[i]
            else:
                param = (min_val + max_val) / 2
            best_params.append(param)
        
        return np.array(best_params), best_fitness
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            'n_qubits': self.n_qubits,
            'n_states': self.n_states,
            'best_state': self.best_state,
            'best_fitness': self.best_fitness,
            'iterations': len(self.history),
            'history': self.history
        }


if __name__ == "__main__":
    # Test quantum-inspired optimizer
    print("⚛️ Testing Quantum-Inspired Optimizer...")
    
    # Simple test function: maximize sum of bits
    def fitness(state: int) -> float:
        # Count number of 1s in binary representation
        return bin(state).count('1')
    
    # Initialize
    qio = QuantumInspiredOptimizer(n_qubits=8, seed=42)
    
    # Optimize
    print("\n🚀 Optimizing...")
    best_state, best_fitness = qio.optimize(fitness, n_iterations=50, n_measurements=10)
    
    print(f"\n🏆 Results:")
    print(f"   Best state: {best_state} (binary: {bin(best_state)})")
    print(f"   Best fitness: {best_fitness}")
    print(f"   Expected max: {qio.n_qubits}")
    
    # Test continuous optimization
    print("\n⚛️ Testing continuous optimization...")
    
    # Sphere function: minimize sum of squares
    def sphere(params: np.ndarray) -> float:
        return -np.sum(params ** 2)  # Negative because we maximize
    
    bounds = [(-5.0, 5.0)] * 5
    
    best_params, best_fitness = qio.optimize_continuous(sphere, bounds, n_iterations=30)
    
    print(f"\n🏆 Results:")
    print(f"   Best params: {best_params}")
    print(f"   Best fitness: {best_fitness:.4f}")
    print(f"   Expected: ~0 (minimum at origin)")
    
    print("\n✅ Quantum-inspired optimizer test complete")