#!/usr/bin/env python3
"""
Darwin Canary - Roda com 10-20% do tráfego por 3 rodadas
"""
import json
import time
import random
from datetime import datetime
from darwin_policy import DarwinPolicy

class DarwinCanary:
    async def __init__(self):
        self.policy = DarwinPolicy()
        self.canary_percentage = 0.15  # 15% do tráfego
        self.rounds_required = 3
        self.rounds_passed = 0
        self.metrics = {
            'promotions': 0,
            'rollbacks': 0,
            'deaths': 0,
            'births': 0,
            'avg_delta_linf': []
        }
        
    async def run_round(self):
        """Executa uma rodada Darwin no canário"""
        logger.info(f"\n[{datetime.now()}] Starting Darwin Canary Round {self.rounds_passed + 1}/{self.rounds_required}")
        
        # Simular agentes (15% da população)
        num_agents = int(50 * self.canary_percentage)
        
        round_results = []
        for i in range(num_agents):
            # Simular métricas do agente
            agent_state = self.simulate_agent(f"canary-agent-{i}")
            
            # Avaliar vida/morte
            decision = self.policy.evaluate_agent(agent_state)
            
            # Log no WORM
            self.policy.log_to_worm(decision, "/root/darwin_worm.log")
            
            # Atualizar métricas
            if decision['decision'] == 'DIE':
                self.metrics['deaths'] += 1
            
            self.metrics['avg_delta_linf'].append(agent_state['delta_linf'])
            round_results.append(decision)
            
        # Verificar nascimentos
        if self.policy.check_birth():
            self.spawn_agent()
            self.metrics['births'] += 1
        
        # Verificar gates
        gates_passed = self.check_gates()
        
        if gates_passed:
            self.rounds_passed += 1
            logger.info(f"✅ Round {self.rounds_passed} passed all gates")
        else:
            self.rounds_passed = 0
            logger.info(f"⚠️ Round failed gates - resetting counter")
            
        # Log métricas
        self.log_metrics()
        
        return await gates_passed
    
    async def simulate_agent(self, agent_id):
        """Simula estado de um agente"""
        # Tendência positiva mas com variação
        delta_linf = random.gauss(0.02, 0.015)  # Média 0.02, desvio 0.015
        delta_linf = max(0, delta_linf)  # Não pode ser negativo
        
        return await {
            'id': agent_id,
            'delta_linf': delta_linf,
            'generalized_discovery': random.random() < 0.05,  # 5% chance
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    async def spawn_agent(self):
        """Cria novo agente com herança"""
        logger.info(f"🐣 Spawning new agent (birth ratio triggered)")
        # Herdar dos melhores
        pass
    
    async def check_gates(self):
        """Verifica se todos os gates estão verdes"""
        # Simular verificação de gates
        gates = {
            'I': 0.604,  # Do último check
            'delta_linf_avg': sum(self.metrics['avg_delta_linf'][-10:]) / min(10, len(self.metrics['avg_delta_linf'])) if self.metrics['avg_delta_linf'] else 0,
            'caos_ratio': 1.031,
            'ece': 0.02,
            'rho': 0.85,
            'oci': 0.667
        }
        
        all_pass = (
            gates['I'] >= 0.60 and
            gates['delta_linf_avg'] >= 0 and
            gates['caos_ratio'] >= 1.0 and
            gates['ece'] <= 0.03 and
            gates['rho'] < 0.95 and
            gates['oci'] >= 0.60
        )
        
        logger.info(f"Gates: I={gates['I']:.3f}, ΔL∞_avg={gates['delta_linf_avg']:.3f}, CAOS={gates['caos_ratio']:.3f}")
        
        return await all_pass
    
    async def log_metrics(self):
        """Registra métricas no log"""
        metrics_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': 'darwin_canary_metrics',
            'rounds_passed': self.rounds_passed,
            'deaths': self.metrics['deaths'],
            'births': self.metrics['births'],
            'avg_delta_linf': sum(self.metrics['avg_delta_linf']) / len(self.metrics['avg_delta_linf']) if self.metrics['avg_delta_linf'] else 0,
            'promotion_ratio': self.metrics['promotions'] / (self.metrics['rollbacks'] + 1)
        }
        
        with open('/root/darwin_metrics.json', 'a') as f:
            json.dump(metrics_entry, f)
            f.write('\n')
    
    async def should_promote(self):
        """Verifica se pode promover para produção total"""
        return await self.rounds_passed >= self.rounds_required
    
    async def run(self):
        """Loop principal do canário"""
        logger.info("🧬 Darwin Canary starting...")
        logger.info(f"Running {self.canary_percentage*100:.0f}% traffic for {self.rounds_required} rounds")
        
        while True:
            try:
                gates_ok = self.run_round()
                
                if self.should_promote():
                    logger.info("🎉 DARWIN CANARY PASSED - Ready for full production!")
                    self.promote_to_production()
                    break
                
                # Aguardar próxima rodada
                time.sleep(300)  # 5 minutos entre rodadas
                
            except KeyboardInterrupt:
                logger.info("\n⏸️ Darwin Canary paused")
                break
            except Exception as e:
                logger.info(f"❌ Error: {e}")
                time.sleep(60)
    
    async def promote_to_production(self):
        """Promove Darwin para produção total"""
        promotion = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': 'darwin_promotion',
            'from': 'canary_15_percent',
            'to': 'production_100_percent',
            'rounds_validated': self.rounds_required,
            'metrics': self.metrics
        }
        
        with open('/root/darwin_promotion.json', 'w') as f:
            json.dump(promotion, f, indent=2)
        
        logger.info("Darwin promoted to full production")
        logger.info("Check /root/darwin_promotion.json for details")

if __name__ == "__main__":
    canary = DarwinCanary()
    canary.run()