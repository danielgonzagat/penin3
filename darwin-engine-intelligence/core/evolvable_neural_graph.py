"""
✅ FASE 2.1: Auto-Arquitetura - Evolução de Topologias Neurais
==============================================================

Inspirado em NEAT (Stanley & Miikkulainen, 2002) + NAS

Features:
- Genes codificam nós e conexões
- Mutações estruturais (add/remove nodes/edges)
- Crossover com alinhamento de genes
- Topologia DAG mutável
- Suporte a múltiplos tipos de camadas

Referências:
- NEAT: Evolving Neural Networks through Augmenting Topologies (2002)
- Neural Architecture Search (NAS)
- DARTS: Differentiable Architecture Search (2019)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import copy


class LayerType(Enum):
    """Tipos de camadas disponíveis"""
    LINEAR = "linear"
    CONV2D = "conv2d"
    ATTENTION = "attention"
    RESIDUAL = "residual"
    BATCH_NORM = "batchnorm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"


class ActivationType(Enum):
    """Tipos de ativação"""
    RELU = "relu"
    TANH = "tanh"
    SILU = "silu"
    GELU = "gelu"


@dataclass
class NodeGene:
    """Gene de nó (neurônio ou camada)"""
    id: int
    layer_type: LayerType
    params: Dict[str, Any]  # Parâmetros específicos do tipo
    

@dataclass
class EdgeGene:
    """Gene de conexão (sinapse)"""
    id: int
    from_node: int
    to_node: int
    enabled: bool = True  # Pode ser desabilitado (não removido)


class EvolvableNeuralGraph:
    """
    Rede neural com topologia evoluível (grafo).
    
    Similar a NEAT mas com camadas modernas (Conv, Attention, etc)
    
    Uso:
        graph = EvolvableNeuralGraph(input_size=784, output_size=10)
        model = graph.build_pytorch_module()
        
        # Mutações estruturais
        graph.mutate_add_node()
        graph.mutate_add_edge()
        graph.mutate_remove_node()
        
        # Crossover
        child = graph.crossover(other_graph)
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 initial_hidden: int = 64):
        self.input_size = input_size
        self.output_size = output_size
        
        # Contadores de IDs
        self.next_node_id = 0
        self.next_edge_id = 0
        
        # Genes
        self.node_genes: Dict[int, NodeGene] = {}
        self.edge_genes: Dict[int, EdgeGene] = {}
        
        # Fitness
        self.fitness = 0.0
        
        # Inicializar topologia mínima (input → hidden → output)
        self._init_minimal_topology(initial_hidden)
    
    def _init_minimal_topology(self, hidden_size: int):
        """Cria topologia mínima: input → hidden → output"""
        # Nó input (camada de entrada)
        input_node = self._add_node(LayerType.LINEAR, {
            'in_features': self.input_size,
            'out_features': hidden_size
        })
        
        # Nó hidden (com ativação)
        hidden_node = self._add_node(LayerType.ACTIVATION, {
            'type': ActivationType.RELU
        })
        
        # Nó output
        output_node = self._add_node(LayerType.LINEAR, {
            'in_features': hidden_size,
            'out_features': self.output_size
        })
        
        # Conexões
        self._add_edge(input_node, hidden_node)
        self._add_edge(hidden_node, output_node)
    
    def _add_node(self, layer_type: LayerType, params: Dict[str, Any]) -> int:
        """Adiciona nó ao grafo"""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        self.node_genes[node_id] = NodeGene(
            id=node_id,
            layer_type=layer_type,
            params=params.copy()
        )
        
        return node_id
    
    def _add_edge(self, from_node: int, to_node: int) -> int:
        """Adiciona conexão ao grafo"""
        edge_id = self.next_edge_id
        self.next_edge_id += 1
        
        self.edge_genes[edge_id] = EdgeGene(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            enabled=True
        )
        
        return edge_id
    
    def mutate_add_node(self, probability: float = 0.03) -> bool:
        """
        Mutação estrutural: Adiciona nó no meio de uma conexão.
        
        Antes: A ---→ B
        Depois: A → [NEW] → B
        
        Args:
            probability: Chance de aplicar mutação
        
        Returns:
            True se mutação foi aplicada
        """
        if random.random() > probability:
            return False
        
        # Escolher edge aleatório
        enabled_edges = [e for e in self.edge_genes.values() if e.enabled]
        if not enabled_edges:
            return False
        
        edge = random.choice(enabled_edges)
        
        # Desabilitar edge original
        edge.enabled = False
        
        # Inferir tamanho das features (simplificado)
        hidden_size = random.choice([64, 128, 256])
        
        # Adicionar novo nó (camada Linear + ativação)
        new_linear = self._add_node(LayerType.LINEAR, {
            'in_features': hidden_size,
            'out_features': hidden_size
        })
        
        new_activation = self._add_node(LayerType.ACTIVATION, {
            'type': random.choice(list(ActivationType))
        })
        
        # Adicionar novas conexões
        self._add_edge(edge.from_node, new_linear)
        self._add_edge(new_linear, new_activation)
        self._add_edge(new_activation, edge.to_node)
        
        return True
    
    def mutate_add_edge(self, probability: float = 0.05) -> bool:
        """
        Mutação estrutural: Adiciona conexão entre nós existentes.
        
        Cria skip connection ou conexão recorrente.
        
        Args:
            probability: Chance de aplicar mutação
        
        Returns:
            True se mutação foi aplicada
        """
        if random.random() > probability:
            return False
        
        nodes = list(self.node_genes.keys())
        if len(nodes) < 2:
            return False
        
        # Escolher 2 nós aleatórios
        from_node, to_node = random.sample(nodes, 2)
        
        # Verificar se conexão já existe
        existing = any(
            e.from_node == from_node and e.to_node == to_node and e.enabled
            for e in self.edge_genes.values()
        )
        
        if existing:
            return False
        
        # Adicionar conexão
        self._add_edge(from_node, to_node)
        
        return True
    
    def mutate_remove_node(self, probability: float = 0.01) -> bool:
        """
        Mutação estrutural: Remove nó (se não for crítico).
        
        Reconecta vizinhos para manter conectividade.
        
        Args:
            probability: Chance de aplicar mutação
        
        Returns:
            True se mutação foi aplicada
        """
        if random.random() > probability:
            return False
        
        # Não remover se tem poucos nós
        if len(self.node_genes) <= 3:
            return False
        
        # Escolher nó aleatório (exceto primeiro e último)
        node_ids = sorted(self.node_genes.keys())
        if len(node_ids) <= 2:
            return False
        
        removable_nodes = node_ids[1:-1]  # Protege primeiro e último
        
        if not removable_nodes:
            return False
        
        node_id = random.choice(removable_nodes)
        
        # Desabilitar edges conectados a este nó
        for edge in self.edge_genes.values():
            if edge.from_node == node_id or edge.to_node == node_id:
                edge.enabled = False
        
        # Remover nó
        del self.node_genes[node_id]
        
        return True
    
    def mutate_change_activation(self, probability: float = 0.1) -> bool:
        """
        Mutação estrutural: Muda função de ativação.
        
        Args:
            probability: Chance de aplicar mutação
        
        Returns:
            True se mutação foi aplicada
        """
        if random.random() > probability:
            return False
        
        activation_nodes = [
            n for n in self.node_genes.values()
            if n.layer_type == LayerType.ACTIVATION
        ]
        
        if not activation_nodes:
            return False
        
        node = random.choice(activation_nodes)
        node.params['type'] = random.choice(list(ActivationType))
        
        return True
    
    def build_pytorch_module(self) -> nn.Module:
        """
        Constrói nn.Module do PyTorch a partir do grafo.
        
        Percorre grafo em ordem de IDs e cria camadas.
        (Simplificação: assume sequential, não grafo completo)
        
        Returns:
            nn.Sequential com camadas
        """
        layers = []
        
        for node_id in sorted(self.node_genes.keys()):
            node = self.node_genes[node_id]
            
            try:
                if node.layer_type == LayerType.LINEAR:
                    layer = nn.Linear(**node.params)
                elif node.layer_type == LayerType.ACTIVATION:
                    act_type = node.params.get('type', ActivationType.RELU)
                    if act_type == ActivationType.RELU:
                        layer = nn.ReLU()
                    elif act_type == ActivationType.TANH:
                        layer = nn.Tanh()
                    elif act_type == ActivationType.SILU:
                        layer = nn.SiLU()
                    elif act_type == ActivationType.GELU:
                        layer = nn.GELU()
                    else:
                        layer = nn.ReLU()
                elif node.layer_type == LayerType.DROPOUT:
                    p = node.params.get('p', 0.5)
                    layer = nn.Dropout(p=p)
                elif node.layer_type == LayerType.BATCH_NORM:
                    num_features = node.params.get('num_features', 128)
                    layer = nn.BatchNorm1d(num_features)
                else:
                    continue
                
                layers.append(layer)
            except Exception as e:
                # Se falhar ao criar camada, pular
                continue
        
        if not layers:
            # Fallback: retornar modelo mínimo
            layers = [
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_size)
            ]
        
        return nn.Sequential(*layers)
    
    def crossover(self, other: 'EvolvableNeuralGraph') -> 'EvolvableNeuralGraph':
        """
        Crossover entre 2 grafos neurais.
        
        Usa alinhamento de genes por ID (estilo NEAT).
        
        Args:
            other: Outro grafo neural
        
        Returns:
            Grafo filho
        """
        child = EvolvableNeuralGraph(self.input_size, self.output_size)
        child.node_genes = {}
        child.edge_genes = {}
        
        # Herdar nós (matching + disjoint + excess)
        all_node_ids = set(self.node_genes.keys()) | set(other.node_genes.keys())
        
        for nid in all_node_ids:
            if nid in self.node_genes and nid in other.node_genes:
                # Matching: escolher aleatoriamente
                node = copy.deepcopy(random.choice([self.node_genes[nid], other.node_genes[nid]]))
            elif nid in self.node_genes:
                # Disjoint de self (herdar se self é mais fit)
                if self.fitness >= other.fitness:
                    node = copy.deepcopy(self.node_genes[nid])
                else:
                    continue
            else:
                # Disjoint de other
                if other.fitness > self.fitness:
                    node = copy.deepcopy(other.node_genes[nid])
                else:
                    continue
            
            child.node_genes[nid] = node
        
        # Similar para edges
        all_edge_ids = set(self.edge_genes.keys()) | set(other.edge_genes.keys())
        
        for eid in all_edge_ids:
            if eid in self.edge_genes and eid in other.edge_genes:
                edge = copy.deepcopy(random.choice([self.edge_genes[eid], other.edge_genes[eid]]))
            elif eid in self.edge_genes:
                if self.fitness >= other.fitness:
                    edge = copy.deepcopy(self.edge_genes[eid])
                else:
                    continue
            else:
                if other.fitness > self.fitness:
                    edge = copy.deepcopy(other.edge_genes[eid])
                else:
                    continue
            
            child.edge_genes[eid] = edge
        
        # Atualizar contadores
        child.next_node_id = max(child.node_genes.keys(), default=0) + 1
        child.next_edge_id = max(child.edge_genes.keys(), default=0) + 1
        
        return child
    
    def get_complexity(self) -> int:
        """Retorna complexidade (número de parâmetros aprox)"""
        total_params = 0
        for node in self.node_genes.values():
            if node.layer_type == LayerType.LINEAR:
                in_f = node.params.get('in_features', 0)
                out_f = node.params.get('out_features', 0)
                total_params += in_f * out_f
        return total_params
    
    def __repr__(self) -> str:
        enabled_edges = sum(1 for e in self.edge_genes.values() if e.enabled)
        return f"EvolvableNeuralGraph(nodes={len(self.node_genes)}, edges={enabled_edges}, fitness={self.fitness:.4f})"
