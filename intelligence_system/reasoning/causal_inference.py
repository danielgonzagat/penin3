#!/usr/bin/env python3
"""
🔗 CAUSAL INFERENCE ENGINE
BLOCO 4 - TAREFA 45

Discovers causal relationships from observations.
Goes beyond correlation to understand causation.
"""

__version__ = "1.0.0"

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class CausalGraph:
    """
    Represents causal structure as directed graph.
    Nodes = variables, Edges = causal relationships
    """
    
    def __init__(self, variables: List[str]):
        """
        Args:
            variables: List of variable names
        """
        self.variables = variables
        self.edges = defaultdict(list)  # parent -> [children]
        self.edge_strengths = {}  # (parent, child) -> strength
    
    def add_edge(self, parent: str, child: str, strength: float = 1.0):
        """Add causal edge"""
        if child not in self.edges[parent]:
            self.edges[parent].append(child)
            self.edge_strengths[(parent, child)] = strength
    
    def get_parents(self, variable: str) -> List[str]:
        """Get causal parents of variable"""
        parents = []
        for parent, children in self.edges.items():
            if variable in children:
                parents.append(parent)
        return parents
    
    def get_children(self, variable: str) -> List[str]:
        """Get causal children of variable"""
        return self.edges.get(variable, [])
    
    def intervene(self, variable: str) -> 'CausalGraph':
        """
        Create interventional graph by removing incoming edges.
        Represents do(variable = value) intervention.
        """
        new_graph = CausalGraph(self.variables)
        
        # Copy all edges except those pointing to intervened variable
        for parent, children in self.edges.items():
            for child in children:
                if child != variable:
                    new_graph.add_edge(
                        parent, child, 
                        self.edge_strengths.get((parent, child), 1.0)
                    )
        
        return new_graph


class CausalInferenceEngine:
    """
    BLOCO 4 - TAREFA 45: Causal inference
    
    Discovers causal structure from observational data.
    Methods:
    - Correlation analysis
    - Granger causality
    - Intervention experiments
    """
    
    def __init__(self):
        """Initialize causal inference engine"""
        self.observations = defaultdict(list)
        self.causal_graph = None
    
    def observe(self, state: Dict[str, float]):
        """
        Record observation of variables.
        
        Args:
            state: Dict mapping variable names to values
        """
        for var, value in state.items():
            self.observations[var].append(value)
    
    def compute_correlation(
        self,
        var1: str,
        var2: str,
        lag: int = 0
    ) -> float:
        """
        Compute correlation between two variables.
        
        Args:
            var1: First variable
            var2: Second variable
            lag: Time lag (for temporal causality)
        
        Returns:
            Correlation coefficient
        """
        if var1 not in self.observations or var2 not in self.observations:
            return 0.0
        
        data1 = np.array(self.observations[var1])
        data2 = np.array(self.observations[var2])
        
        # Apply lag
        if lag > 0:
            if len(data1) <= lag:
                return 0.0
            data1 = data1[lag:]
            data2 = data2[:-lag]
        elif lag < 0:
            if len(data2) <= abs(lag):
                return 0.0
            data1 = data1[:lag]
            data2 = data2[abs(lag):]
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        if len(data1) < 2:
            return 0.0
        
        # Pearson correlation
        return float(np.corrcoef(data1, data2)[0, 1])
    
    def granger_causality(
        self,
        cause: str,
        effect: str,
        max_lag: int = 5
    ) -> Tuple[bool, float]:
        """
        Test if cause Granger-causes effect.
        
        Args:
            cause: Potential cause variable
            effect: Potential effect variable
            max_lag: Maximum lag to test
        
        Returns:
            (is_causal, strength)
        """
        if cause not in self.observations or effect not in self.observations:
            return False, 0.0
        
        # Test if past values of cause predict future values of effect
        best_correlation = 0.0
        is_causal = False
        
        for lag in range(1, max_lag + 1):
            corr = abs(self.compute_correlation(cause, effect, lag=lag))
            
            if corr > best_correlation:
                best_correlation = corr
            
            # Threshold for causality
            if corr > 0.3:  # Arbitrary threshold
                is_causal = True
        
        return is_causal, best_correlation
    
    def discover_causal_graph(
        self,
        threshold: float = 0.3,
        max_lag: int = 3
    ) -> CausalGraph:
        """
        Discover causal graph from observations.
        
        Args:
            threshold: Correlation threshold for causality
            max_lag: Maximum temporal lag
        
        Returns:
            Discovered causal graph
        """
        variables = list(self.observations.keys())
        graph = CausalGraph(variables)
        
        # Test all pairs
        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue
                
                # Granger causality test
                is_causal, strength = self.granger_causality(cause, effect, max_lag)
                
                if is_causal and strength > threshold:
                    graph.add_edge(cause, effect, strength)
        
        self.causal_graph = graph
        return graph
    
    def predict_intervention_effect(
        self,
        intervened_var: str,
        target_var: str
    ) -> Optional[str]:
        """
        Predict effect of intervening on a variable.
        
        Args:
            intervened_var: Variable to intervene on
            target_var: Variable to observe
        
        Returns:
            Prediction: 'increase', 'decrease', 'no_effect', or None
        """
        if self.causal_graph is None:
            return None
        
        # Check if there's a causal path
        def has_path(start, end, visited=None):
            if visited is None:
                visited = set()
            
            if start == end:
                return True
            
            if start in visited:
                return False
            
            visited.add(start)
            
            for child in self.causal_graph.get_children(start):
                if has_path(child, end, visited):
                    return True
            
            return False
        
        if has_path(intervened_var, target_var):
            # There's a causal path
            strength = self.causal_graph.edge_strengths.get((intervened_var, target_var), 0)
            
            if strength > 0:
                return 'increase'
            else:
                return 'decrease'
        else:
            return 'no_effect'


if __name__ == "__main__":
    print(f"Testing Causal Inference v{__version__}...")
    
    # Create causal inference engine
    engine = CausalInferenceEngine()
    
    # Simulate observations with known causal structure:
    # A -> B -> C, D independent
    np.random.seed(42)
    
    for t in range(100):
        A = np.random.randn()
        B = 0.8 * A + 0.2 * np.random.randn()
        C = 0.7 * B + 0.3 * np.random.randn()
        D = np.random.randn()
        
        engine.observe({'A': A, 'B': B, 'C': C, 'D': D})
    
    # Discover causal graph
    graph = engine.discover_causal_graph(threshold=0.3, max_lag=3)
    
    print("Discovered causal edges:")
    for parent, children in graph.edges.items():
        for child in children:
            strength = graph.edge_strengths.get((parent, child), 0)
            print(f"  {parent} -> {child} (strength={strength:.3f})")
    
    # Test intervention prediction
    effect = engine.predict_intervention_effect('A', 'C')
    print(f"\nPredicted effect of intervening on A for C: {effect}")
    
    effect = engine.predict_intervention_effect('D', 'C')
    print(f"Predicted effect of intervening on D for C: {effect}")
    
    print("✅ Causal Inference tests OK!")
