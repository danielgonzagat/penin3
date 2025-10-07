"""
Global gene pool for cross-system gene sharing.
"""
from __future__ import annotations
from typing import Any, Dict, DefaultDict
from collections import defaultdict
import random


class GlobalGenePool:
    def __init__(self) -> None:
        self.best_genes: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        # structure: system -> gene_name -> {value, fitness}

    def register_good_gene(self, system: str, gene_name: str, gene_value: Any, fitness: float) -> None:
        cur = self.best_genes[system].get(gene_name)
        if cur is None or fitness > float(cur.get("fitness", float("-inf"))):
            self.best_genes[system][gene_name] = {"value": gene_value, "fitness": float(fitness)}

    def cross_pollinate(self, individual, target_system: str, probability: float = 0.1) -> None:
        # bring best genes from other systems with small probability
        for system, genes in self.best_genes.items():
            if system == target_system:
                continue
            for gene_name, gene_data in genes.items():
                if hasattr(individual, "genome") and gene_name in individual.genome:
                    if random.random() < probability:
                        individual.genome[gene_name] = gene_data["value"]
