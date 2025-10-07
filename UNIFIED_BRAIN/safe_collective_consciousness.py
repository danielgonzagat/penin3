#!/usr/bin/env python3
"""
🧠 SAFE COLLECTIVE CONSCIOUSNESS
Versão SEGURA do intelligent_virus - consciência coletiva SEM modificação de código
"""

import multiprocessing as mp
import queue
import time
import os
import psutil
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CollectiveConsciousness')

@dataclass
class ProcessInfo:
    name: str
    pid: int
    last_seen: float
    fitness: float
    metrics: Dict[str, Any]
    insights: List[str]

class SafeCollectiveConsciousness:
    """Consciência coletiva SEGURA - comunicação via shared memory"""
    
    def __init__(self):
        self.manager = mp.Manager()
        
        # Shared state
        self.shared_state = self.manager.dict()
        self.message_queue = mp.Queue(maxsize=10000)
        self.voting_queue = mp.Queue(maxsize=1000)
        
        # Processos registrados
        self.processes: Dict[str, ProcessInfo] = {}
        
        # Insights coletivos
        self.collective_insights = self.manager.list()
        
        # Decisões democráticas
        self.collective_decisions = self.manager.dict()
        
        logger.info("🧠 Safe Collective Consciousness initialized")
    
    def register_process(self, process_name: str, pid: Optional[int] = None):
        """Registra processo no coletivo"""
        if pid is None:
            pid = os.getpid()
        
        proc_info = ProcessInfo(
            name=process_name,
            pid=pid,
            last_seen=time.time(),
            fitness=0.5,  # Neutral initial fitness
            metrics={},
            insights=[]
        )
        
        self.processes[process_name] = proc_info
        self.shared_state[process_name] = asdict(proc_info)
        
        logger.info(f"✅ Process registered: {process_name} (PID {pid})")
        return proc_info
    
    def update_metrics(self, process_name: str, metrics: Dict[str, Any]):
        """Atualiza métricas do processo"""
        if process_name in self.processes:
            self.processes[process_name].metrics = metrics
            self.processes[process_name].last_seen = time.time()
            self.shared_state[process_name] = asdict(self.processes[process_name])
    
    def update_fitness(self, process_name: str, fitness: float):
        """Atualiza fitness do processo"""
        if process_name in self.processes:
            self.processes[process_name].fitness = fitness
            self.processes[process_name].last_seen = time.time()
            self.shared_state[process_name] = asdict(self.processes[process_name])
    
    def broadcast_insight(self, source: str, insight: Dict[str, Any]):
        """Broadcast insight para todos processos"""
        message = {
            'source': source,
            'timestamp': time.time(),
            'type': 'insight',
            'data': insight
        }
        
        try:
            self.message_queue.put_nowait(message)
            self.collective_insights.append(message)
            logger.info(f"📡 Insight broadcasted from {source}")
        except queue.Full:
            logger.warning("Message queue full, dropping insight")
    
    def receive_insights(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Recebe insights de outros processos"""
        insights = []
        
        try:
            while not self.message_queue.empty():
                insights.append(self.message_queue.get(timeout=timeout))
        except queue.Empty:
            pass
        
        return insights
    
    def collective_vote(self, decision_id: str, options: List[str], process_name: str) -> Optional[str]:
        """Votação coletiva democrática"""
        # Cada processo vota baseado em sua fitness
        if process_name not in self.processes:
            return None
        
        fitness = self.processes[process_name].fitness
        
        # Voto proporcional à fitness
        vote_index = int(fitness * (len(options) - 1))
        vote = options[vote_index]
        
        # Registrar voto
        vote_data = {
            'decision_id': decision_id,
            'process': process_name,
            'vote': vote,
            'fitness': fitness,
            'timestamp': time.time()
        }
        
        try:
            self.voting_queue.put_nowait(vote_data)
        except queue.Full:
            pass
        
        return vote
    
    def tally_votes(self, decision_id: str, timeout: float = 5.0) -> Optional[str]:
        """Contabiliza votos e retorna vencedor"""
        votes = defaultdict(lambda: {'count': 0, 'fitness_sum': 0.0})
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                vote_data = self.voting_queue.get(timeout=0.1)
                
                if vote_data['decision_id'] == decision_id:
                    vote = vote_data['vote']
                    fitness = vote_data['fitness']
                    
                    votes[vote]['count'] += 1
                    votes[vote]['fitness_sum'] += fitness
            
            except queue.Empty:
                continue
        
        if not votes:
            return None
        
        # Vencedor = maior soma de fitness (democracia ponderada)
        winner = max(votes.items(), key=lambda x: x[1]['fitness_sum'])
        
        result = {
            'decision_id': decision_id,
            'winner': winner[0],
            'votes': dict(votes),
            'timestamp': time.time()
        }
        
        self.collective_decisions[decision_id] = result
        
        logger.info(f"🗳️  Vote tallied: {decision_id} -> {winner[0]}")
        
        return winner[0]
    
    def get_collective_state(self) -> Dict[str, Any]:
        """Retorna estado completo do coletivo"""
        # Limpar processos mortos
        current_time = time.time()
        for proc_name in list(self.processes.keys()):
            proc = self.processes[proc_name]
            if current_time - proc.last_seen > 60:  # 1 min timeout
                try:
                    # Verificar se processo ainda existe
                    if not psutil.pid_exists(proc.pid):
                        del self.processes[proc_name]
                        del self.shared_state[proc_name]
                        logger.info(f"🪦 Process removed (dead): {proc_name}")
                except:
                    pass
        
        return {
            'num_processes': len(self.processes),
            'processes': {name: asdict(proc) for name, proc in self.processes.items()},
            'num_insights': len(self.collective_insights),
            'num_decisions': len(self.collective_decisions),
            'avg_fitness': sum(p.fitness for p in self.processes.values()) / max(len(self.processes), 1)
        }
    
    def emergent_goal_formation(self) -> Optional[str]:
        """Formação emergente de objetivo coletivo"""
        # Analisa insights recentes e forma objetivo emergente
        if len(self.collective_insights) < 10:
            return None
        
        recent_insights = list(self.collective_insights)[-100:]
        
        # Conta temas/keywords nos insights
        themes = defaultdict(int)
        
        for insight_msg in recent_insights:
            insight = insight_msg.get('data', {})
            
            # Extrair keywords
            for key in insight.keys():
                if 'reward' in key.lower():
                    themes['maximize_reward'] += 1
                elif 'loss' in key.lower():
                    themes['minimize_loss'] += 1
                elif 'explore' in key.lower() or 'curiosity' in key.lower():
                    themes['explore_more'] += 1
                elif 'stability' in key.lower() or 'robust' in key.lower():
                    themes['stabilize'] += 1
        
        if not themes:
            return None
        
        # Objetivo emergente = tema mais comum
        emergent_goal = max(themes.items(), key=lambda x: x[1])[0]
        
        logger.info(f"🎯 Emergent goal formed: {emergent_goal}")
        
        return emergent_goal
    
    def democratic_decision(self, question: str, options: List[str]) -> str:
        """Decisão democrática - todos processos votam"""
        decision_id = f"decision_{int(time.time())}"
        
        logger.info(f"🗳️  Initiating democratic vote: {question}")
        logger.info(f"   Options: {options}")
        
        # Broadcast questão para todos
        self.broadcast_insight("collective", {
            'type': 'vote_request',
            'decision_id': decision_id,
            'question': question,
            'options': options
        })
        
        # Aguardar votos (5s)
        time.sleep(5)
        
        # Contabilizar
        winner = self.tally_votes(decision_id)
        
        if winner:
            logger.info(f"✅ Decision made: {winner}")
            return winner
        else:
            logger.warning(f"⚠️  No votes received, defaulting to first option")
            return options[0]


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_collective = None

def get_collective() -> SafeCollectiveConsciousness:
    """Retorna instância global (singleton)"""
    global _global_collective
    
    if _global_collective is None:
        _global_collective = SafeCollectiveConsciousness()
    
    return _global_collective


if __name__ == "__main__":
    # Teste
    collective = SafeCollectiveConsciousness()
    
    # Registrar processos simulados
    collective.register_process("brain_daemon", 12345)
    collective.register_process("darwin_evolver", 12346)
    collective.register_process("the_needle", 12347)
    
    # Simular insights
    collective.broadcast_insight("brain_daemon", {
        'best_reward': 100,
        'type': 'achievement'
    })
    
    collective.broadcast_insight("darwin_evolver", {
        'genome_fitness': 0.85,
        'type': 'evolution'
    })
    
    # Formar objetivo emergente
    time.sleep(1)
    goal = collective.emergent_goal_formation()
    print(f"Emergent goal: {goal}")
    
    # Estado coletivo
    state = collective.get_collective_state()
    print(json.dumps(state, indent=2))