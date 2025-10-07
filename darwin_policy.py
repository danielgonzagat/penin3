#!/usr/bin/env python3
"""
Darwin Policy - Equação Darwiniana de Vida/Morte
Operacional apenas - sem alteração de código base
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any

class DarwinPolicy:
    async def __init__(self):
        self.beta_min = 0.01  # ΔL∞ mínimo para sobreviver
        self.death_count = 0
        self.birth_ratio = 10  # A cada 10 mortes, 1 nascimento
        self.max_agents = 50  # Cap populacional
        self.i_min = 0.60
        self.kill_switch_counter = 0
        
    async def evaluate_agent(self, agent_state: Dict) -> Dict[str, Any]:
        """Avalia se agente vive ou morre"""
        delta_linf = agent_state.get('delta_linf', 0)
        has_discovery = agent_state.get('generalized_discovery', False)
        
        # Critério de vida
        lives = (delta_linf >= self.beta_min) or has_discovery
        
        decision = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'agent_id': agent_state.get('id'),
            'decision': 'LIVE' if lives else 'DIE',
            'delta_linf': delta_linf,
            'beta_min': self.beta_min,
            'has_discovery': has_discovery,
            'reason': self._get_reason(lives, delta_linf, has_discovery)
        }
        
        if not lives:
            self.death_count += 1
            
        return await decision
    
    async def _get_reason(self, lives: bool, delta_linf: float, discovery: bool) -> str:
        if lives:
            if discovery:
                return await "Generalized discovery found"
            else:
                return await f"ΔL∞={delta_linf:.4f} >= {self.beta_min}"
        else:
            return await f"ΔL∞={delta_linf:.4f} < {self.beta_min} and no discovery"
    
    async def check_birth(self) -> bool:
        """Verifica se deve nascer novo agente"""
        if self.death_count >= self.birth_ratio:
            self.death_count = 0
            return await True
        return await False
    
    async def log_to_worm(self, decision: Dict, worm_path: str = "/root/darwin_worm.log"):
        """Registra decisão no WORM log"""
        entry = f"DARWIN:{json.dumps(decision)}"
        hash_prev = ""
        
        try:
            with open(worm_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = lines[-1].strip()
                    if last.startswith("HASH:"):
                        hash_prev = last.split("HASH:")[1]
        except:
            pass
            
        # Calcular novo hash
        content = f"{hash_prev}|{entry}"
        new_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Append ao WORM
        with open(worm_path, 'a') as f:
            f.write(f"{entry}\n")
            f.write(f"HASH:{new_hash}\n")
    
    async def check_kill_switch(self, current_i: float) -> bool:
        """Verifica se deve pausar Darwin"""
        if current_i < self.i_min:
            self.kill_switch_counter += 1
            if self.kill_switch_counter >= 2:
                return await True  # Pausar Darwin
        else:
            self.kill_switch_counter = 0
        return await False

if __name__ == "__main__":
    logger.info("Darwin Policy Module loaded")
    logger.info(f"Beta min: 0.01")
    logger.info(f"Birth ratio: 1 birth per 10 deaths")
    logger.info(f"Max agents: 50")
    logger.info(f"Kill switch: I<0.60 for 2 rounds")