#!/usr/bin/env python3
"""
🎛️ META-CONTROLLER ATIVO
Controla parâmetros do sistema baseado em performance real
"""

import torch
from collections import deque
from datetime import datetime
from brain_logger import brain_logger

class MetaController:
    """
    Controla LR, exploration, e outros hyperparams baseado em tendências
    """
    def __init__(self, intervention_threshold: float = 0.05):
        self.performance_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.interventions = []
        self.intervention_threshold = intervention_threshold
        
        # Limites de segurança
        self.lr_min = 1e-5
        self.lr_max = 1e-2
        self.temp_min = 0.3
        self.temp_max = 3.0
    
    def analyze_and_act(self, stats: dict, router, optimizer):
        """
        Analisa tendências e intervém se necessário
        ✅ CORREÇÃO #8: API consistente - recebe router diretamente
        
        Args:
            stats: dict com metrics do daemon
            router: AdaptiveRouter instance (com temperature, top_k, etc)
            optimizer: torch optimizer
        """
        # Atualiza histórico
        avg_reward = stats.get('avg_reward_last_100', 0)
        avg_loss = stats.get('avg_loss', 0)
        
        self.performance_history.append(avg_reward)
        self.loss_history.append(avg_loss)
        
        if len(self.performance_history) < 10:
            return  # Precisa de histórico
        
        # Calcula tendências
        recent_perf = list(self.performance_history)[-10:]
        recent_loss = list(self.loss_history)[-10:]
        
        perf_trend = (recent_perf[-1] - recent_perf[0]) / max(1e-6, abs(recent_perf[0]))
        loss_trend = (recent_loss[-1] - recent_loss[0]) / max(1e-6, abs(recent_loss[0]))
        
        # INTERVENÇÃO 1: Stagnação de performance
        if abs(perf_trend) < self.intervention_threshold:
            self._intervene_stagnation(optimizer, router, perf_trend, stats)
        
        # INTERVENÇÃO 2: Loss explodindo
        elif loss_trend > 0.5:
            self._intervene_exploding_loss(optimizer, router, stats)
        
        # INTERVENÇÃO 3: Loss muito baixa mas reward não melhora
        elif avg_loss < 0.1 and perf_trend < 0:
            self._intervene_overfitting(optimizer, router, stats)
        
        # INTERVENÇÃO 4: Reward caindo
        elif perf_trend < -0.2:
            self._intervene_degradation(optimizer, router, stats)
    
    def _intervene_stagnation(self, optimizer, router, trend, stats):
        """Performance estagnada - aumenta exploration"""
        intervention = {
            'timestamp': datetime.now().isoformat(),
            'type': 'stagnation',
            'trend': trend,
            'actions': []
        }
        
        # Aumenta LR temporariamente
        for g in optimizer.param_groups:
            old_lr = g['lr']
            g['lr'] = min(self.lr_max, g['lr'] * 1.3)
            intervention['actions'].append(f"LR: {old_lr:.6f} → {g['lr']:.6f}")
        
        # ✅ CORREÇÃO #8: Usa router diretamente
        if router is not None:
            old_temp = router.temperature
            router.temperature = min(self.temp_max, router.temperature * 1.15)
            intervention['actions'].append(f"Temp: {old_temp:.3f} → {router.temperature:.3f}")
            
            old_k = router.top_k
            router.top_k = min(router.max_k, router.top_k + 2)
            intervention['actions'].append(f"top_k: {old_k} → {router.top_k}")
        
        self.interventions.append(intervention)
        brain_logger.warning(f"🎛️ META: Stagnation detected (trend={trend:.4f}), increasing exploration")
    
    def _intervene_exploding_loss(self, optimizer, router, stats):
        """Loss explodindo - reduz LR"""
        intervention = {
            'timestamp': datetime.now().isoformat(),
            'type': 'exploding_loss',
            'loss': stats.get('avg_loss', 0),
            'actions': []
        }
        
        # Reduz LR drasticamente
        for g in optimizer.param_groups:
            old_lr = g['lr']
            g['lr'] = max(self.lr_min, g['lr'] * 0.5)
            intervention['actions'].append(f"LR: {old_lr:.6f} → {g['lr']:.6f}")
        
        # ✅ CORREÇÃO #8: Usa router diretamente
        if router is not None:
            old_temp = router.temperature
            router.temperature = max(self.temp_min, router.temperature * 0.8)
            intervention['actions'].append(f"Temp: {old_temp:.3f} → {router.temperature:.3f}")
        
        self.interventions.append(intervention)
        brain_logger.error(f"🎛️ META: Loss exploding! Reducing LR and temperature")
    
    def _intervene_overfitting(self, optimizer, router, stats):
        """Overfitting - aumenta regularização"""
        intervention = {
            'timestamp': datetime.now().isoformat(),
            'type': 'overfitting',
            'actions': []
        }
        
        # ✅ CORREÇÃO #8: Usa router diretamente
        if router is not None:
            old_temp = router.temperature
            router.temperature = min(self.temp_max, router.temperature * 1.2)
            intervention['actions'].append(f"Temp: {old_temp:.3f} → {router.temperature:.3f}")
        
        self.interventions.append(intervention)
        brain_logger.warning(f"🎛️ META: Overfitting detected, increasing exploration")
    
    def _intervene_degradation(self, optimizer, router, stats):
        """Performance caindo - rollback parcial"""
        intervention = {
            'timestamp': datetime.now().isoformat(),
            'type': 'degradation',
            'actions': []
        }
        
        # Reduz LR para estabilizar
        for g in optimizer.param_groups:
            old_lr = g['lr']
            g['lr'] = max(self.lr_min, g['lr'] * 0.7)
            intervention['actions'].append(f"LR: {old_lr:.6f} → {g['lr']:.6f}")
        
        self.interventions.append(intervention)
        brain_logger.error(f"🎛️ META: Performance degrading! Stabilizing system")
    
    def get_stats(self):
        """Retorna stats do meta-controller"""
        return {
            'total_interventions': len(self.interventions),
            'interventions_by_type': self._count_by_type(),
            'recent_interventions': self.interventions[-5:] if self.interventions else []
        }
    
    def _count_by_type(self):
        counts = {}
        for i in self.interventions:
            t = i['type']
            counts[t] = counts.get(t, 0) + 1
        return counts


if __name__ == "__main__":
    print("🎛️ Meta-Controller Module")
    
    # Test
    mc = MetaController()
    
    # Simulate stagnation
    for i in range(15):
        mc.performance_history.append(10.0 + (i % 3) * 0.1)
        mc.loss_history.append(0.5)
    
    class DummyBrain:
        alpha = 0.85
        router = None
    
    class DummyOpt:
        param_groups = [{'lr': 3e-4}]
    
    brain = DummyBrain()
    opt = DummyOpt()
    
    mc.analyze_and_act({'avg_reward_last_100': 10.0, 'avg_loss': 0.5}, brain, opt)
    
    print(f"Interventions: {mc.get_stats()['total_interventions']}")
    print("✅ Meta-Controller OK")
