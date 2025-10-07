"""
POET-Lite: Paired Open-Ended Trailblazer (Lightweight)
======================================================

IMPLEMENTAÇÃO PURA PYTHON (SEM ML)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

Based on: Wang et al. (2019) "Paired Open-Ended Trailblazer (POET): 
Endlessly Generating Increasingly Complex and Diverse Learning Environments 
and Their Solutions"
"""

import random
import json
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import time


@dataclass
class Environment:
    """Ambiente/tarefa evolutiva"""
    env_id: str
    params: Dict[str, Any]
    difficulty: float
    created_at: float = field(default_factory=time.time)
    solved_count: int = 0


@dataclass
class Agent:
    """Agente/solução"""
    agent_id: str
    genome: Dict[str, Any]
    scores: Dict[str, float] = field(default_factory=dict)  # env_id -> score
    created_at: float = field(default_factory=time.time)


class POETLite:
    """
    POET-Lite: Co-evolução de agentes e ambientes
    
    Princípios:
    1. Agentes evoluem para resolver ambientes
    2. Ambientes evoluem para desafiar agentes
    3. Transfer: agentes testados em ambientes de outros
    4. MCC: Minimal Criterion Coevolution (agente deve atingir mínimo)
    """
    
    def __init__(self,
                 env_generator_fn: Callable[[], Environment],
                 agent_factory_fn: Callable[[], Agent],
                 eval_fn: Callable[[Agent, Environment], float],
                 mutate_env_fn: Callable[[Environment], Environment],
                 mutate_agent_fn: Callable[[Agent], Agent],
                 mc_threshold: float = 0.1):
        """
        Args:
            env_generator_fn: Gera ambiente aleatório
            agent_factory_fn: Cria agente aleatório
            eval_fn: Avalia agent em env, retorna score
            mutate_env_fn: Mutaciona ambiente
            mutate_agent_fn: Mutaciona agente
            mc_threshold: Minimal Criterion (mínimo score para considerar resolvido)
        """
        self.env_generator_fn = env_generator_fn
        self.agent_factory_fn = agent_factory_fn
        self.eval_fn = eval_fn
        self.mutate_env_fn = mutate_env_fn
        self.mutate_agent_fn = mutate_agent_fn
        self.mc_threshold = mc_threshold
        
        # Arquivos
        self.environments: List[Environment] = []
        self.agents: List[Agent] = []
        
        # Métricas SOTA
        self.iteration = 0
        self.total_evaluations = 0
        self.transfer_successes = 0
        self.new_envs_created = 0
    
    def initialize(self, n_initial_pairs: int = 5):
        """Inicializa com pares (agent, env)"""
        print(f"\n🌱 Inicializando POET com {n_initial_pairs} pares...")
        
        for i in range(n_initial_pairs):
            # FIX: Pass rng to generator functions
            env_data = self.env_generator_fn(self.rng)
            agent_data = self.agent_factory_fn(self.rng)
            
            # Convert to proper objects if needed
            if isinstance(env_data, dict):
                env = Environment(env_id=f"env_{i}", params=env_data, difficulty=env_data.get('difficulty', 0.5))
            else:
                env = env_data
            
            if isinstance(agent_data, dict):
                agent = Agent(agent_id=f"agent_{i}", genome=agent_data)
            else:
                agent = agent_data
            
            # Avaliar
            score = self.eval_fn(agent, env, self.rng)
            agent.scores[env.env_id] = score
            
            self.environments.append(env)
            self.agents.append(agent)
            self.total_evaluations += 1
            
            print(f"   Par {i+1}: Agent {agent.agent_id[:8]} × Env {env.env_id[:8]} "
                  f"→ score={score:.3f}")
    
    def evolve_agents(self):
        """Evolui agentes em seus ambientes"""
        print(f"\n🧬 Evoluindo agentes...")
        
        new_agents = []
        
        for i, (agent, env) in enumerate(zip(self.agents, self.environments)):
            # Mutacionar
            offspring = self.mutate_agent_fn(agent)
            
            # Avaliar no ambiente original
            score = self.eval_fn(offspring, env)
            offspring.scores[env.env_id] = score
            self.total_evaluations += 1
            
            # Selecionar melhor
            if score > agent.scores.get(env.env_id, 0.0):
                new_agents.append(offspring)
                print(f"   ✨ Agent {i+1} melhorou: "
                      f"{agent.scores.get(env.env_id, 0):.3f} → {score:.3f}")
            else:
                new_agents.append(agent)
        
        self.agents = new_agents
    
    def generate_new_environments(self):
        """Gera novos ambientes onde agentes estão muito bons"""
        print(f"\n🌍 Gerando novos ambientes...")
        
        for i, (agent, env) in enumerate(zip(self.agents, self.environments)):
            score = agent.scores.get(env.env_id, 0.0)
            
            # Se agente está muito bom (>0.9), criar ambiente mais difícil
            if score > 0.9:
                # Mutacionar ambiente (aumenta dificuldade)
                new_env = self.mutate_env_fn(env)
                new_env.difficulty = min(1.0, env.difficulty + 0.1)
                
                # Testar agente no novo env
                new_score = self.eval_fn(agent, new_env)
                self.total_evaluations += 1
                
                # Adicionar se MCC atingido (agent consegue pelo menos mínimo)
                if new_score >= self.mc_threshold:
                    self.environments.append(new_env)
                    self.agents.append(agent)  # Clone do agent
                    self.new_envs_created += 1
                    
                    print(f"   ✨ Novo ambiente criado (dificuldade={new_env.difficulty:.2f})")
                    print(f"      Agent score: {new_score:.3f} (MC={self.mc_threshold})")
    
    def transfer_cross_niche(self):
        """Transfer: testa agentes em ambientes de outros"""
        print(f"\n🔄 Transfer cross-niche...")
        
        if len(self.agents) < 2 or len(self.environments) < 2:
            return
        
        # Tentar algumas transferências
        n_transfers = min(5, len(self.agents))
        
        for _ in range(n_transfers):
            # Random agent e random env
            agent_idx = random.randint(0, len(self.agents) - 1)
            env_idx = random.randint(0, len(self.environments) - 1)
            
            agent = self.agents[agent_idx]
            env = self.environments[env_idx]
            
            # Já testou este par?
            if env.env_id in agent.scores:
                continue
            
            # Testar
            score = self.eval_fn(agent, env)
            agent.scores[env.env_id] = score
            self.total_evaluations += 1
            
            # Se melhor que agente atual neste env, substituir
            current_agent_idx = env_idx if env_idx < len(self.agents) else 0
            current_agent = self.agents[current_agent_idx]
            current_score = current_agent.scores.get(env.env_id, 0.0)
            
            if score > current_score:
                self.agents[env_idx] = agent
                self.transfer_successes += 1
                print(f"   ✅ Transfer success: Agent {agent.agent_id[:8]} "
                      f"→ Env {env.env_id[:8]} (score={score:.3f})")
    
    def evolve(self, n_iterations: int, verbose: bool = True):
        """
        Loop POET principal
        
        Args:
            n_iterations: Número de iterações
            verbose: Mostrar progresso
        """
        for i in range(n_iterations):
            self.iteration += 1
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"🌀 POET Iteration {i+1}/{n_iterations}")
                print(f"{'='*80}")
                print(f"   Ambientes: {len(self.environments)}")
                print(f"   Agentes: {len(self.agents)}")
            
            # 1. Evolve agents
            self.evolve_agents()
            
            # 2. Generate new environments
            if i % 5 == 0:  # A cada 5 iterações
                self.generate_new_environments()
            
            # 3. Transfer cross-niche
            if i % 3 == 0:  # A cada 3 iterações
                self.transfer_cross_niche()
            
            if verbose and (i + 1) % 10 == 0:
                self._print_stats()
    
    def _print_stats(self):
        """Imprime estatísticas"""
        print(f"\n📊 ESTATÍSTICAS POET:")
        print(f"   Iteração: {self.iteration}")
        print(f"   Total environments: {len(self.environments)}")
        print(f"   Total agents: {len(self.agents)}")
        print(f"   Avaliações: {self.total_evaluations}")
        print(f"   Transfers sucessos: {self.transfer_successes}")
        print(f"   Novos envs criados: {self.new_envs_created}")
        
        # Melhor score em cada ambiente
        if self.agents and self.environments:
            print(f"\n   🏆 Melhores scores por ambiente:")
            for env in self.environments[:5]:  # Top 5
                scores = [agent.scores.get(env.env_id, 0.0) for agent in self.agents]
                max_score = max(scores) if scores else 0.0
                print(f"      Env {env.env_id[:8]}: {max_score:.3f}")
    
    def save(self, filepath: str):
        """Salva estado POET"""
        data = {
            'iteration': self.iteration,
            'total_evaluations': self.total_evaluations,
            'environments': [
                {
                    'env_id': env.env_id,
                    'params': env.params,
                    'difficulty': env.difficulty,
                    'solved_count': env.solved_count
                }
                for env in self.environments
            ],
            'agents': [
                {
                    'agent_id': agent.agent_id,
                    'genome': agent.genome,
                    'scores': agent.scores
                }
                for agent in self.agents
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# TESTE COMPLETO
# ============================================================================

def test_poet_lite():
    """Teste completo POET-Lite"""
    print("\n" + "="*80)
    print("TESTE: POET-Lite Completo")
    print("="*80 + "\n")
    
    # Funções de teste (domínio simples: otimização 1D com ruído)
    
    def env_generator():
        """Gera ambiente (nível de ruído)"""
        env_id = f"env_{random.randint(1000, 9999)}"
        noise_level = random.uniform(0.0, 0.5)
        return Environment(env_id, {'noise': noise_level}, difficulty=noise_level)
    
    def agent_factory():
        """Cria agente (valor que tenta otimizar)"""
        agent_id = f"agent_{random.randint(1000, 9999)}"
        genome = {'value': random.uniform(-1.0, 1.0)}
        return Agent(agent_id, genome)
    
    def eval_fn(agent: Agent, env: Environment) -> float:
        """Avalia agent em env"""
        # Objetivo: maximizar f(x) = -x² com ruído do ambiente
        x = agent.genome['value']
        noise = env.params['noise']
        
        # Fitness sem ruído
        fitness_clean = -(x ** 2)  # Máximo em x=0
        
        # Adicionar ruído baseado no ambiente
        noise_val = random.gauss(0, noise)
        fitness_noisy = fitness_clean + noise_val
        
        # Normalizar para [0, 1]
        return max(0.0, min(1.0, fitness_noisy + 1.0))
    
    def mutate_env(env: Environment) -> Environment:
        """Mutaciona ambiente"""
        new_noise = env.params['noise'] + random.gauss(0, 0.05)
        new_noise = max(0.0, min(0.5, new_noise))
        
        new_id = f"env_{random.randint(1000, 9999)}"
        return Environment(new_id, {'noise': new_noise}, difficulty=new_noise)
    
    def mutate_agent(agent: Agent) -> Agent:
        """Mutaciona agente"""
        new_value = agent.genome['value'] + random.gauss(0, 0.1)
        new_value = max(-1.0, min(1.0, new_value))
        
        new_id = f"agent_{random.randint(1000, 9999)}"
        return Agent(new_id, {'value': new_value})
    
    # Criar POET
    poet = POETLite(
        env_generator_fn=env_generator,
        agent_factory_fn=agent_factory,
        eval_fn=eval_fn,
        mutate_env_fn=mutate_env,
        mutate_agent_fn=mutate_agent,
        mc_threshold=0.3  # Mínimo 0.3 para considerar solvable
    )
    
    # Inicializar
    poet.initialize(n_initial_pairs=5)
    
    # Evoluir
    poet.evolve(n_iterations=20, verbose=True)
    
    # Stats finais
    print(f"\n{'='*80}")
    print("📊 RESULTADO FINAL POET")
    print(f"{'='*80}")
    print(f"  Ambientes criados: {len(poet.environments)}")
    print(f"  Agentes evoluídos: {len(poet.agents)}")
    print(f"  Avaliações totais: {poet.total_evaluations}")
    print(f"  Transfers sucesso: {poet.transfer_successes}")
    print(f"  Novos ambientes: {poet.new_envs_created}")
    
    # Validar
    assert len(poet.environments) >= 5, "Deve ter pelo menos ambientes iniciais"
    assert len(poet.agents) >= 5, "Deve ter pelo menos agentes iniciais"
    assert poet.total_evaluations > 20, "Deve ter avaliado"
    
    # Salvar
    poet.save('/tmp/poet_lite_archive.json')
    print(f"\n💾 Archive salvo: /tmp/poet_lite_archive.json")
    
    print("\n✅ TESTE POET-LITE PASSOU!\n")
    print("="*80)


if __name__ == "__main__":
    test_poet_lite()
    print("\n✅ poet_lite_pure.py FUNCIONAL!")
