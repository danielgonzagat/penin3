#!/usr/bin/env python3
"""
Darwinacci-Ω - Núcleo Evolutivo
Sistema de evolução contínua com aprendizado de máquina
"""
import json, time, random, logging, threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass
class EvolutionState:
    generation: int
    fitness_score: float
    mutation_rate: float
    crossover_rate: float
    population_size: int
    best_individual: Dict[str, Any]
    population: List[Dict[str, Any]]
    metrics: Dict[str, float]

@dataclass
class LearningData:
    input_features: List[float]
    output_target: float
    context: Dict[str, Any]
    timestamp: str

class DarwinacciOmega:
    """Darwinacci-Ω - Núcleo evolutivo com aprendizado"""
    
    def __init__(self, state_path: str = "/root/qwen_darwinacci_state.json"):
        self.state_path = state_path
        self.evolution_state: Optional[EvolutionState] = None
        self.learning_data: List[LearningData] = []
        self.logger = logging.getLogger(__name__)
        
        # Parâmetros evolutivos
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        # Carregar estado existente
        self._load_state()
        
        # Thread de evolução contínua
        self.evolution_thread = None
        self.running = False
        
    def _load_state(self):
        """Carrega estado evolutivo existente"""
        if Path(self.state_path).exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.evolution_state = EvolutionState(**data)
                self.logger.info(f"Estado evolutivo carregado: Geração {self.evolution_state.generation}")
            except Exception as e:
                self.logger.error(f"Erro ao carregar estado: {e}")
                self._initialize_state()
        else:
            self._initialize_state()
    
    def _initialize_state(self):
        """Inicializa estado evolutivo"""
        self.evolution_state = EvolutionState(
            generation=0,
            fitness_score=0.0,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            population_size=self.population_size,
            best_individual={},
            population=[],
            metrics={}
        )
        self._save_state()
        self.logger.info("Estado evolutivo inicializado")
    
    def _save_state(self):
        """Salva estado evolutivo"""
        try:
            with open(self.state_path, 'w') as f:
                json.dump(asdict(self.evolution_state), f, indent=2)
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")
    
    def _generate_individual(self) -> Dict[str, Any]:
        """Gera indivíduo aleatório"""
        return {
            "id": f"ind_{int(time.time() * 1000000)}",
            "genes": {
                "command_preference": random.uniform(0, 1),
                "safety_threshold": random.uniform(0.1, 0.9),
                "efficiency_weight": random.uniform(0, 1),
                "innovation_rate": random.uniform(0, 0.5),
                "exploration_rate": random.uniform(0, 1)
            },
            "fitness": 0.0,
            "generation": self.evolution_state.generation,
            "created_at": datetime.now().isoformat()
        }
    
    def _evaluate_fitness(self, individual: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Avalia fitness do indivíduo"""
        genes = individual["genes"]
        
        # Fitness baseado em múltiplos critérios
        fitness_components = {
            "efficiency": genes["efficiency_weight"],
            "safety": 1.0 - abs(genes["safety_threshold"] - 0.7),  # Preferir threshold moderado
            "innovation": genes["innovation_rate"],
            "exploration": genes["exploration_rate"],
            "command_preference": genes["command_preference"]
        }
        
        # Peso dos componentes
        weights = {
            "efficiency": 0.3,
            "safety": 0.3,
            "innovation": 0.2,
            "exploration": 0.1,
            "command_preference": 0.1
        }
        
        # Calcular fitness ponderado
        fitness = sum(fitness_components[key] * weights[key] for key in fitness_components)
        
        # Ajustar baseado no contexto
        if context.get("success_rate", 0) > 0.8:
            fitness *= 1.1  # Bonus por alta taxa de sucesso
        
        if context.get("error_rate", 0) > 0.2:
            fitness *= 0.9  # Penalidade por alta taxa de erro
        
        return min(1.0, max(0.0, fitness))
    
    def _mutate_individual(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica mutação ao indivíduo"""
        mutated = individual.copy()
        genes = mutated["genes"].copy()
        
        for gene_name in genes:
            if random.random() < self.mutation_rate:
                # Mutação gaussiana
                noise = random.gauss(0, 0.1)
                genes[gene_name] = max(0, min(1, genes[gene_name] + noise))
        
        mutated["genes"] = genes
        mutated["id"] = f"ind_{int(time.time() * 1000000)}"
        mutated["generation"] = self.evolution_state.generation
        
        return mutated
    
    def _crossover_individuals(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza crossover entre dois indivíduos"""
        child = {
            "id": f"ind_{int(time.time() * 1000000)}",
            "genes": {},
            "fitness": 0.0,
            "generation": self.evolution_state.generation,
            "created_at": datetime.now().isoformat()
        }
        
        # Crossover uniforme
        for gene_name in parent1["genes"]:
            if random.random() < 0.5:
                child["genes"][gene_name] = parent1["genes"][gene_name]
            else:
                child["genes"][gene_name] = parent2["genes"][gene_name]
        
        return child
    
    def _evolve_generation(self):
        """Evolui uma geração"""
        if not self.evolution_state.population:
            # Inicializar população
            self.evolution_state.population = [self._generate_individual() for _ in range(self.population_size)]
        
        # Avaliar fitness da população atual
        for individual in self.evolution_state.population:
            context = {"success_rate": random.uniform(0.7, 0.9), "error_rate": random.uniform(0.1, 0.3)}
            individual["fitness"] = self._evaluate_fitness(individual, context)
        
        # Ordenar por fitness
        self.evolution_state.population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Atualizar melhor indivíduo
        self.evolution_state.best_individual = self.evolution_state.population[0].copy()
        self.evolution_state.fitness_score = self.evolution_state.best_individual["fitness"]
        
        # Criar nova geração
        new_population = []
        
        # Elitismo - manter melhores indivíduos
        elite = self.evolution_state.population[:self.elite_size]
        new_population.extend(elite)
        
        # Gerar descendentes
        while len(new_population) < self.population_size:
            # Seleção por torneio
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            if random.random() < self.crossover_rate:
                child = self._crossover_individuals(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Aplicar mutação
            child = self._mutate_individual(child)
            new_population.append(child)
        
        self.evolution_state.population = new_population
        self.evolution_state.generation += 1
        
        # Atualizar métricas
        self._update_metrics()
        
        # Salvar estado
        self._save_state()
        
        self.logger.info(f"Geração {self.evolution_state.generation} evoluída. Fitness: {self.evolution_state.fitness_score:.3f}")
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Seleção por torneio"""
        tournament = random.sample(self.evolution_state.population, min(tournament_size, len(self.evolution_state.population)))
        return max(tournament, key=lambda x: x["fitness"])
    
    def _update_metrics(self):
        """Atualiza métricas evolutivas"""
        if not self.evolution_state.population:
            return
        
        fitness_values = [ind["fitness"] for ind in self.evolution_state.population]
        
        self.evolution_state.metrics = {
            "avg_fitness": np.mean(fitness_values),
            "max_fitness": np.max(fitness_values),
            "min_fitness": np.min(fitness_values),
            "std_fitness": np.std(fitness_values),
            "diversity": self._calculate_diversity(),
            "convergence_rate": self._calculate_convergence_rate()
        }
    
    def _calculate_diversity(self) -> float:
        """Calcula diversidade da população"""
        if len(self.evolution_state.population) < 2:
            return 0.0
        
        # Calcular distância média entre indivíduos
        distances = []
        for i in range(len(self.evolution_state.population)):
            for j in range(i + 1, len(self.evolution_state.population)):
                dist = self._individual_distance(
                    self.evolution_state.population[i],
                    self.evolution_state.population[j]
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _individual_distance(self, ind1: Dict[str, Any], ind2: Dict[str, Any]) -> float:
        """Calcula distância entre dois indivíduos"""
        genes1 = ind1["genes"]
        genes2 = ind2["genes"]
        
        distances = []
        for gene_name in genes1:
            if gene_name in genes2:
                dist = abs(genes1[gene_name] - genes2[gene_name])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_convergence_rate(self) -> float:
        """Calcula taxa de convergência"""
        if self.evolution_state.generation < 2:
            return 0.0
        
        # Simular cálculo de convergência
        return min(1.0, self.evolution_state.generation / 100.0)
    
    def add_learning_data(self, input_features: List[float], output_target: float, context: Dict[str, Any]):
        """Adiciona dados de aprendizado"""
        learning_entry = LearningData(
            input_features=input_features,
            output_target=output_target,
            context=context,
            timestamp=datetime.now().isoformat()
        )
        
        self.learning_data.append(learning_entry)
        
        # Manter apenas os últimos 1000 registros
        if len(self.learning_data) > 1000:
            self.learning_data = self.learning_data[-1000:]
        
        self.logger.info(f"Dados de aprendizado adicionados. Total: {len(self.learning_data)}")
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Retorna parâmetros otimizados do melhor indivíduo"""
        if not self.evolution_state.best_individual:
            return {
                "command_preference": 0.5,
                "safety_threshold": 0.7,
                "efficiency_weight": 0.5,
                "innovation_rate": 0.1,
                "exploration_rate": 0.5
            }
        
        return self.evolution_state.best_individual["genes"]
    
    def start_evolution(self):
        """Inicia evolução contínua"""
        if self.evolution_thread and self.evolution_thread.is_alive():
            return
        
        self.running = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()
        self.logger.info("Evolução contínua iniciada")
    
    def stop_evolution(self):
        """Para evolução contínua"""
        self.running = False
        if self.evolution_thread:
            self.evolution_thread.join(timeout=5)
        self.logger.info("Evolução contínua parada")
    
    def _evolution_loop(self):
        """Loop principal de evolução"""
        while self.running:
            try:
                self._evolve_generation()
                time.sleep(60)  # Evoluir a cada minuto
            except Exception as e:
                self.logger.error(f"Erro na evolução: {e}")
                time.sleep(10)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas evolutivas"""
        if not self.evolution_state:
            return {"status": "not_initialized"}
        
        return {
            "generation": self.evolution_state.generation,
            "fitness_score": self.evolution_state.fitness_score,
            "population_size": self.evolution_state.population_size,
            "metrics": self.evolution_state.metrics,
            "best_individual": self.evolution_state.best_individual,
            "learning_data_count": len(self.learning_data),
            "running": self.running
        }

if __name__ == "__main__":
    # Teste do Darwinacci-Ω
    darwinacci = DarwinacciOmega()
    
    # Adicionar dados de aprendizado
    darwinacci.add_learning_data(
        input_features=[0.8, 0.6, 0.9, 0.7],
        output_target=0.85,
        context={"success": True, "efficiency": 0.8}
    )
    
    # Evoluir algumas gerações
    for i in range(5):
        darwinacci._evolve_generation()
        print(f"Geração {darwinacci.evolution_state.generation}: Fitness = {darwinacci.evolution_state.fitness_score:.3f}")
    
    # Mostrar estatísticas
    stats = darwinacci.get_statistics()
    print(f"Estatísticas: {json.dumps(stats, indent=2)}")
    
    # Mostrar parâmetros otimizados
    optimal_params = darwinacci.get_optimal_parameters()
    print(f"Parâmetros otimizados: {json.dumps(optimal_params, indent=2)}")
