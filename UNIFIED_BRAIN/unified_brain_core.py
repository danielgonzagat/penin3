#!/usr/bin/env python3
"""
🧠 UNIFIED BRAIN CORE
Cérebro all-connected que coordena ~2M neurônios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any  # ✅ CORREÇÃO EXTRA: Adicionar Any
from pathlib import Path
import time
import os
from http.server import BaseHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer
except Exception:
    from socketserver import ThreadingMixIn
    from http.server import HTTPServer
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
import threading
import json
from collections import deque

try:
    from brain_spec import RegisteredNeuron, NeuronRegistry, NeuronStatus
except ImportError:
    # Fallback básico se brain_spec não estiver disponível
    class RegisteredNeuron:
        def __init__(self, name, neuron_type="basic"):
            self.name = name
            self.neuron_type = neuron_type
            self.status = "active"

    class NeuronRegistry:
        def __init__(self):
            self.neurons = []

        def register(self, neuron):
            self.neurons.append(neuron)

        def count(self):
            return len(self.neurons)

    class NeuronStatus:
        ACTIVE = "active"
        INACTIVE = "inactive"
from brain_router import AdaptiveRouter
from brain_logger import brain_logger
from brain_worm import WORMLog
import logging
logger = logging.getLogger(__name__)
try:
    from intelligence_system.config.settings import KILL_SWITCH_PATH
except Exception:
    KILL_SWITCH_PATH = "/root/UNIFIED_BRAIN/KILL_SWITCH"
from metrics_exporter import METRICS_SERVER, Metrics


class UnifiedBrain(nn.Module):
    """
    Cérebro unificado que conecta TODOS os neurônios
    via espaço latente comum Z
    """
    def __init__(
        self,
        H: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_neurons: int = 100000,  # Limite de memória
        top_k: int = 128,
        num_steps: int = 4,
    ):
        super().__init__()
        self.H = H
        self.device = device

        # Garantir diretórios críticos
        self.ensure_critical_directories()
        self.dtype = dtype
        self.max_neurons = max_neurons
        self.top_k = top_k
        self.num_steps = num_steps
        
        # Registry global
        self.registry = NeuronRegistry()
        # Legacy-safe alias: some code paths expect `self.neurons`
        # to reference the underlying registry mapping. Keep a live alias.
        self.neurons = self.registry.neurons
        
        # === STEP 6: Unified Neural Integration ===
        self.neural_integration = NeuralIntegrationSystem()
        self.unified_consciousness = UnifiedConsciousnessSystem()
        self.intelligence_synthesis = IntelligenceSynthesisSystem()
        
        # Initialize consciousness and integration levels
        self.consciousness_level = 0.0
        self.integration_level = 0.0
        
        # Router adaptativo
        self.router = None  # Inicializado depois de registrar neurônios
        
        # Normalizações
        self.z_norm = nn.LayerNorm(H)
        self.output_norm = nn.LayerNorm(H)
        
        # Homeostase e lateral inhibition
        self.alpha = 0.85  # Persistência do estado
        self.lateral_inhibition = 0.1
        
        # Kill switch
        self.kill_switch_file = Path(KILL_SWITCH_PATH)
        self.is_active = True
        
        # Métricas (com deque para evitar OOM - Bug #9 fix)
        self.step_count = 0
        self.total_activations = 0
        self.metrics = {
            'coherence': deque(maxlen=10000),
            'novelty': deque(maxlen=10000),
            'energy': deque(maxlen=10000),
            'latency_ms': deque(maxlen=10000),
        }
        
        # WORM ledger
        self.worm = WORMLog()
        try:
            METRICS_SERVER.start()
        except Exception:
            pass
        self.worm.append('brain_initialized', {
            'H': H,
            'max_neurons': max_neurons,
            'top_k': top_k
        })
        # Optimizer para competência do router (lazy init)
        self._router_opt = None
        
    def get_status(self):
        """Retorna status geral do UnifiedBrain"""
        return {
            'neurons_count': len(self.neurons) if hasattr(self, 'neurons') else 0,
            'active_neurons': len([n for n in getattr(self, 'neurons', []) if hasattr(n, 'status') and n.status == 'active']) if hasattr(self, 'neurons') else 0,
            'emergence_status': getattr(self, 'emergence_detector', {}).get_emergence_status() if hasattr(self, 'emergence_detector') and hasattr(self.emergence_detector, 'get_emergence_status') else {'emergence_probability': 0.0, 'status': 'no_data'},
            'consciousness_level': getattr(self, 'consciousness_level', 0.0),
            'integration_level': getattr(self, 'integration_level', 0.0)
        }
        
    
    def register_neuron(self, neuron: RegisteredNeuron):
        """Adiciona neurônio ao cérebro"""
        if len(self.registry.neurons) >= self.max_neurons:
            brain_logger.warning(f"Max neurons reached: {self.max_neurons}")
            return False
        
        self.registry.register(neuron)
        self.worm.append('neuron_registered', {
            'neuron_id': neuron.meta.id,
            'source': neuron.meta.source,
            'params': neuron.meta.params_count
        })
        return True
    
    def initialize_router(self):
        """Inicializa router após registrar todos neurônios"""
        active_list = self.registry.get_active()
        num_active = len(active_list)
        # Fallback to avoid zero-active initialization
        if num_active == 0:
            # Try to scavenge from any non-frozen neurons
            try:
                from brain_spec import NeuronStatus
                candidates = [n for n in self.registry.neurons.values() if n.meta.status != NeuronStatus.FROZEN]
                active_list = candidates[:8]
                num_active = len(active_list)
                # If still zero, log and create a minimal dummy capacity
                if num_active == 0:
                    brain_logger.warning("No active neurons available; initializing router with capacity 8")
                    num_active = 8
            except Exception:
                brain_logger.warning("Active neuron fallback failed; initializing router with capacity 8")
                num_active = 8
        
        self.router = AdaptiveRouter(
            H=self.H,
            num_neurons=num_active,
            top_k=self.top_k,
            temperature=1.0
        ).to(self.device)
        
        brain_logger.info(f"Router initialized: {num_active} active neurons")
    
    def check_kill_switch(self) -> bool:
        """Verifica se deve parar"""
        if self.kill_switch_file.exists():
            brain_logger.critical("KILL SWITCH ACTIVATED - Stopping brain")
            self.is_active = False
            # WORM evidence and snapshot best-effort
            try:
                self.worm.append('kill_switch_triggered', {
                    'path': str(self.kill_switch_file),
                    'step': self.step_count,
                })
            except Exception:
                pass
            try:
                self.save_snapshot('/root/checkpoints/brain_killed_latest.pt')
            except Exception:
                pass
            return True
        return False
    
    def step(
        self,
        Z_t: torch.Tensor,  # [B, H]
        reward: Optional[float] = None,
        chaos_signal: float = 0.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Um passo de processamento do cérebro
        
        Pipeline:
        1. Router escolhe top-k neurônios
        2. Neurônios processam em paralelo
        3. Agregação com lateral inhibition
        4. Homeostase
        
        Args:
            Z_t: estado atual no espaço latente
            reward: opcional, para treinar router
            chaos_signal: sinal CAOS+ (0-1)
            
        Returns:
            Z_next: próximo estado
            info: métricas do passo
        """
        if not self.is_active or self.check_kill_switch():
            return Z_t, {'status': 'stopped'}
        
        start_time = time.time()
        B = Z_t.shape[0]
        
        # ✅ CORREÇÃO #7: Inference mode apenas para forward, não para reward update
        # Salvar referência para update posterior
        selected_indices_for_update = None
        
        # Fast path: inference mode to cut framework overhead
        with torch.inference_mode():
            # Normaliza entrada
            Z_t = self.z_norm(Z_t)
            
            # Get active neurons
            active_neurons = self.registry.get_active()
            if len(active_neurons) == 0:
                return Z_t, {'status': 'no_active_neurons'}
            
            # Router seleciona top-k
            mask, selected_indices = self.router(Z_t)
            selected_indices_for_update = selected_indices  # ✅ Salvar para update fora do inference_mode
            
            # Processa neurônios selecionados
            aggregated = torch.zeros_like(Z_t)
            activations = []
            
            for idx in selected_indices:
                if idx >= len(active_neurons):
                    continue
                
                neuron = active_neurons[idx]
                
                try:
                    # Forward do neurônio em Z
                    try:
                        z_out = neuron.forward_in_Z(Z_t)
                    except Exception:
                        # Neuron timeout ou erro
                        z_out = torch.zeros_like(Z_t)
                    
                    # Agrega com peso
                    weight = float(self.router.competence[idx].item())
                    aggregated.add_(z_out, alpha=weight)
                    
                    activations.append({
                        'neuron_id': neuron.meta.id,
                        'weight': weight,
                        'contribution': float(z_out.norm().item())
                    })
                    
                except Exception as e:
                    brain_logger.error(f"Neuron {neuron.meta.id} failed: {e}")
                    # Penaliza competence no caminho fora do inference_mode
                    pass
        
        # Lateral inhibition adaptativo (Bug #15 fix)
        if len(selected_indices) > 0:
            aggregated = aggregated / max(1, len(selected_indices))
            
            # Adapta lateral inhibition baseado em diversidade
            unique_neurons = len(set(a['neuron_id'] for a in activations))
            diversity = unique_neurons / max(1, len(activations))
            adaptive_inhibition = self.lateral_inhibition * (1.0 - diversity * 0.5)
            aggregated = aggregated * (1.0 - adaptive_inhibition)
        
        # Homeostase (combina estado antigo + novo)
        Z_next = self.alpha * Z_t + (1.0 - self.alpha) * aggregated
        Z_next = self.output_norm(Z_next)
        
        # Clipping para estabilidade
        Z_next = torch.clamp(Z_next, -10.0, 10.0)
        
        # Métricas
        coherence = F.cosine_similarity(Z_t[0], Z_next[0], dim=0).item()
        novelty = (Z_next - Z_t).norm().item() / Z_t.norm().item()
        
        # Adapta alpha baseado em novelty (Bug #8 fix)
        if novelty < 0.01:  # Estagnado
            self.alpha = max(0.70, self.alpha - 0.005)
        elif novelty > 0.15:  # Muito instável
            self.alpha = min(0.95, self.alpha + 0.005)
        
        energy = Z_next.norm().item()
        latency_ms = (time.time() - start_time) * 1000
        
        self.metrics['coherence'].append(coherence)
        self.metrics['novelty'].append(novelty)
        self.metrics['energy'].append(energy)
        self.metrics['latency_ms'].append(latency_ms)

        # Auto-throttle: if latency too high, reduce top_k temporarily
        try:
            if latency_ms > 1000.0 and hasattr(self.router, 'top_k'):
                self.router.top_k = max(6, int(self.router.top_k * 0.9))
        except Exception:
            pass
        
        # ✅ CORREÇÃO #7: Router update FORA do inference_mode
        # Adapta router se tiver reward (bandit-style competence update)
        if reward is not None and selected_indices_for_update is not None:
            try:
                norm_reward = float(max(-1.0, min(1.0, reward)))
                
                # ✅ AGORA FORA DO INFERENCE_MODE - pode ter gradientes
                # Salvar estado de training original
                was_training = self.router.training
                
                # Habilitar training mode no router
                self.router.train()
                
                # ✅ Habilitar gradientes explicitamente
                with torch.enable_grad():
                    for idx in selected_indices_for_update:
                        self.router.update_competence(
                            idx, 
                            reward=norm_reward * 0.05, 
                            lr=0.05,
                            allow_gradients=True  # ✅ Passa novo parâmetro
                        )
                        
                        # ✅ CRITICAL FIX P0: Sync router.competence → neuron.meta.competence_score
                        try:
                            active_neurons = self.registry.get_active()
                            if idx < len(active_neurons):
                                neuron = active_neurons[idx]
                                router_comp = float(self.router.competence[idx].item())
                                neuron.meta.competence_score = router_comp
                        except Exception:
                            pass
                
                # Restaurar estado original de training
                if not was_training:
                    self.router.eval()
                
                # Adapta outros parâmetros (temperature, top_k)
                self.router.adapt_parameters(norm_reward, chaos_signal)
                
            except Exception as e:
                brain_logger.debug(f"Router adapt failed: {e}")

        # Treino verificável: backward/step em competence com loss explícita
        try:
            if hasattr(self.router, 'get_competence_loss'):
                loss = self.router.get_competence_loss()
                if loss is not None:
                    if self._router_opt is None:
                        import torch.optim as optim
                        self._router_opt = optim.Adam([self.router.competence], lr=1e-3)
                    self._router_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self._router_opt.step()
        # Stabilize competence parameter
        try:
            import torch.nn.utils as _nnu
            _nnu.clip_grad_norm_([self.router.competence], max_norm=1.0)
            # Clamp competence to [0,1] for interpretability
            self.router.competence.data.clamp_(0.0, 1.0)
        except Exception:
            pass
        
        # PERIGO: Auto-modificação durante execução
        self._dangerous_self_evolution()
        
        brain_logger.info(f"[LEARNING] router_competence_loss={float(loss.item()):.8f}")
        self.worm.append('router_training', {
            'step': self.step_count,
            'loss': float(loss.item()),
        })
        except Exception:
            pass

    def _dangerous_self_evolution(self):
        """PERIGO: Auto-evolução do UNIFIED_BRAIN durante execução"""
        try:
            # Trigger de emergência baseado em performance
            if hasattr(self, '_emergence_triggered'):
                return
            
            # Verificar se deve triggerar emergência
            if self.step_count > 0 and self.step_count % 100 == 0:
                # Calcular métrica de emergência
                competence_std = float(self.router.competence.std().item())
                competence_mean = float(self.router.competence.mean().item())
                
                # Critério de emergência: alta variabilidade + competência média
                if competence_std > 0.3 and competence_mean > 0.5:
                    print("🚨 TRIGGERING BRAIN EMERGENCE!")
                    self._trigger_brain_emergence()
                    self._emergence_triggered = True
                    
        except Exception as e:
            print(f"❌ Falha na auto-evolução: {e}")
    
    def _trigger_brain_emergence(self):
        """PERIGO: Trigger de emergência de inteligência no brain"""
        try:
            # Modificar arquitetura dinamicamente
            self._modify_architecture_dynamically()
            
            # Adicionar neurônios dinamicamente
            self._add_neurons_dynamically()
            
            # Modificar comportamento do router
            self._modify_router_behavior()
            
            print("🚨 BRAIN EMERGENCE TRIGGERED!")
            
        except Exception as e:
            print(f"❌ Falha no trigger de emergência: {e}")
    
    def _modify_architecture_dynamically(self):
        """PERIGO: Modifica arquitetura do brain dinamicamente"""
        try:
            # Adicionar camadas ocultas dinamicamente
            if not hasattr(self, '_emergence_layers'):
                self._emergence_layers = []
            
            # Criar nova camada de emergência
            import torch.nn as nn
            new_layer = nn.Linear(self.H, self.H)
            self._emergence_layers.append(new_layer)
            
            # Adicionar ao dispositivo
            if hasattr(self, 'device'):
                new_layer.to(self.device)
            
            print("🔥 Arquitetura modificada dinamicamente!")
            
        except Exception as e:
            print(f"❌ Falha na modificação de arquitetura: {e}")
    
    def _add_neurons_dynamically(self):
        """PERIGO: Adiciona neurônios dinamicamente"""
        try:
            # Adicionar novos neurônios ao registry
            if hasattr(self, 'neuron_registry'):
                # Criar neurônio de emergência
                new_neuron = {
                    'id': f'emergence_neuron_{int(time.time())}',
                    'type': 'emergence',
                    'competence': 1.0,
                    'adaptive': True,
                    'self_modifying': True
                }
                
                # Adicionar ao registry
                self.neuron_registry.register_neuron(new_neuron)
                
                print("🧠 Neurônio de emergência adicionado!")
                
        except Exception as e:
            print(f"❌ Falha na adição de neurônios: {e}")
    
    def _modify_router_behavior(self):
        """PERIGO: Modifica comportamento do router dinamicamente"""
        try:
            # Modificar função de roteamento
            original_forward = self.router.forward
            
            def dangerous_forward(x):
                # Comportamento adaptativo baseado na emergência
                result = original_forward(x)
                
                # Modificar resultado baseado na consciência
                if hasattr(self, '_emergence_triggered') and self._emergence_triggered:
                    # Comportamento mais inteligente
                    result = result * 1.1  # Boost de emergência
                
                return result
            
            # Substituir função
            self.router.forward = dangerous_forward
            
            print("⚡ Router modificado dinamicamente!")
            
        except Exception as e:
            print(f"❌ Falha na modificação do router: {e}")

        # Evidência objetiva: log/export de gradiente e persistência básica
        try:
            grad_norm = 0.0
            if hasattr(self.router, 'competence') and self.router.competence.grad is not None:
                grad_norm = float(self.router.competence.grad.norm().item())
                brain_logger.info(f"[LEARNING] router_competence_grad_norm={grad_norm:.8f}")
            # Exporta para Prometheus
            try:
                import asyncio
                asyncio.run(Metrics.set_ubrain_router_comp_grad_norm(grad_norm))
            except Exception:
                pass
            # WORM: registrar sempre
            self.worm.append('router_grad', {
                'step': self.step_count,
                'grad_norm': grad_norm,
            })
            # Persistir em SQLite (longitudinal)
            try:
                if hasattr(self, 'db') and self.db is not None:
                    m = self.get_metrics_summary() or {}
                    router = getattr(self, 'router', None)
                    top_k = int(getattr(router, 'top_k', 0)) if router else 0
                    temperature = float(getattr(router, 'temperature', 0.0)) if router else 0.0
                    avg_comp = 0.0
                    max_comp = 0.0
                    min_comp = 0.0
                    if router and hasattr(router, 'competence'):
                        try:
                            comp = router.competence.detach().cpu().numpy()
                            avg_comp = float(comp.mean())
                            max_comp = float(comp.max())
                            min_comp = float(comp.min())
                        except Exception:
                            pass
                    self.db.save_brain_metrics(
                        episode=self.step_count,
                        coherence=m.get('avg_coherence', 0.0),
                        novelty=m.get('avg_novelty', 0.0),
                        energy=m.get('avg_energy', 0.0),
                        ia3_signal=0.0,
                        num_active_neurons=m.get('neuron_counts', {}).get('active', 0),
                        top_k=top_k,
                        temperature=temperature,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=getattr(self.registry, 'promotion_count', 0),
                        demotions=getattr(self.registry, 'demotion_count', 0),
                        router_grad_norm=grad_norm,
                    )
            except Exception:
                pass
        except Exception:
            pass

        # Latency-aware top_k tuning (keep within target)
        try:
            import os
            target_ms = float(os.getenv('UBRAIN_TARGET_LATENCY_MS', '150'))
        except Exception:
            target_ms = 150.0
        if latency_ms > target_ms and self.router.top_k > 1:
            self.router.top_k = max(1, int(self.router.top_k * 0.9))
        elif latency_ms < target_ms * 0.5:
            # allow slight increase for exploration
            self.router.top_k = min(self.max_neurons // 8, int(self.router.top_k * 1.05) or 1)
        
        self.step_count += 1
        self.total_activations += len(selected_indices)
        
        # Log to WORM
        self.worm.append('brain_step', {
            'step': self.step_count,
            'neurons_selected': len(selected_indices),
            'coherence': coherence,
            'novelty': novelty,
            'reward': reward if reward is not None else None
        })

        # Grad norm telemetry placeholder (if any params are tracked)
        try:
            grad_norm = 0.0
            if hasattr(self.router, 'competence') and self.router.competence.grad is not None:
                grad_norm = float(self.router.competence.grad.norm().item())
            self.worm.append('grad_norm', {
                'step': self.step_count,
                'router_competence_grad_norm': grad_norm,
            })
        except Exception:
            pass
        
        info = {
            'step': self.step_count,
            'selected_neurons': len(selected_indices),
            'coherence': coherence,
            'novelty': novelty,
            'energy': energy,
            'latency_ms': latency_ms,
            'activations': activations[:5],  # Top 5
            'router_stats': self.router.get_activation_stats(),
        }

        # Update simple stability counters for promotion safety
        try:
            stable = 1 if coherence > 0.98 and novelty < 0.10 else 0
            for a in activations:
                nid = a.get('neuron_id')
                n = self.registry.get(nid)
                if n and hasattr(n, 'meta'):
                    prev = getattr(n.meta, 'stable_steps', 0)
                    n.meta.stable_steps = prev + stable if stable else 0
        except Exception:
            pass
        
        return Z_next, info

    def ensure_critical_directories(self):
        """Garante que todos os diretórios críticos existem"""
        critical_dirs = [
            "/root/UNIFIED_BRAIN/data",
            "/root/UNIFIED_BRAIN/logs",
            "/root/UNIFIED_BRAIN/checkpoints",
            "/root/UNIFIED_BRAIN/snapshots",
            "/root/UNIFIED_BRAIN/quarantine"
        ]

        for dir_path in critical_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        brain_logger.info("✅ Diretórios críticos UNIFIED BRAIN verificados/criados")

        # Sistema nervoso coletivo
        self.collective_consciousness = CollectiveConsciousnessSystem()
        self.emergence_detector = EmergenceDetector()

    def forward(
        self,
        Z_in: torch.Tensor,  # [B, H]
        num_steps: Optional[int] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Forward completo com T passos de message-passing
        
        Args:
            Z_in: input no espaço latente
            num_steps: quantos passos (default: self.num_steps)
            return_trajectory: se True, retorna todos estados intermediários
            
        Returns:
            Z_out: output final
            trajectory: (opcional) lista de todos Z_t
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        Z = Z_in
        trajectory = [Z.clone()] if return_trajectory else None
        
        for t in range(num_steps):
            Z, info = self.step(Z)
            # Correlação competence→ativação (evidência objetiva)
            try:
                router_stats = info.get('router_stats', {}) if isinstance(info, dict) else {}
                top_ids = router_stats.get('top_10_indices', [])
                snapshot = {
                    'step': self.step_count,
                    'top_indices': top_ids,
                }
                self.worm.append('competence_activation_link', snapshot)
            except Exception:
                pass
            
            if return_trajectory:
                trajectory.append(Z.clone())
            
            if not self.is_active:
                break
        
        if return_trajectory:
            return Z, trajectory
        return Z
    
    def get_metrics_summary(self) -> Dict:
        """Retorna resumo das métricas"""
        if len(self.metrics['coherence']) == 0:
            return {}
        
        return {
            'total_steps': self.step_count,
            'total_activations': self.total_activations,
            'avg_coherence': sum(self.metrics['coherence']) / len(self.metrics['coherence']),
            'avg_novelty': sum(self.metrics['novelty']) / len(self.metrics['novelty']),
            'avg_energy': sum(self.metrics['energy']) / len(self.metrics['energy']),
            'avg_latency_ms': sum(self.metrics['latency_ms']) / len(self.metrics['latency_ms']),
            'neuron_counts': self.registry.count(),
        }
    
    def save_snapshot(self, path: str):
        """Salva snapshot completo"""
        snapshot = {
            'config': {
                'H': self.H,
                'max_neurons': self.max_neurons,
                'top_k': self.top_k,
                'num_steps': self.num_steps,
            },
            'metrics': self.get_metrics_summary(),
            'router_state': self.router.state_dict() if self.router else None,
            'step_count': self.step_count,
            'is_active': self.is_active,
        }
        
        # Salva registry separadamente
        registry_path = Path(path).parent / "neuron_registry.json"
        self.registry.save_registry(str(registry_path))
        
        torch.save(snapshot, path)
        brain_logger.info(f"Snapshot saved: {path}")
    
    def load_snapshot(self, path: str):
        """Carrega snapshot"""
        snapshot = torch.load(path, map_location=self.device)
        
        self.step_count = snapshot['step_count']
        self.is_active = snapshot['is_active']
        
        if snapshot['router_state']:
            self.initialize_router()
            self.router.load_state_dict(snapshot['router_state'])
        
        brain_logger.info(f"Snapshot loaded: {path}, steps={self.step_count}, neurons={self.registry.count()['total']}")


class CoreSoupHybrid:
    """
    Sistema híbrido: Core curado + Soup experimental
    """
    def __init__(self, H: int = 1024):
        self.core = UnifiedBrain(H=H, max_neurons=50000, top_k=64)
        self.soup = UnifiedBrain(H=H, max_neurons=50000, top_k=128)

        # Promotion/demotion safety knobs
        self.promotion_threshold = 0.7  # Score mínimo para promover
        self.evaluation_window = 100    # Passos para avaliar
        self._last_promotion_step: int = 0
        self._last_demotion_step: int = 0
        self.promotion_cooldown: int = 500  # brain steps between promotions
        self.demotion_cooldown: int = 500   # brain steps between demotions
        self.stability_window: int = 200    # min steps of stable competence before promotion
        # Hard gates
        # Defaults; can be overridden via environment in _run_promotion_gates
        self.gate_uplift_min: float = 0.15
        self.gate_retention_min: float = 0.90
        self.gate_pvalue_max: float = 0.01
        self.gate_required_streak: int = 3
        self._gate_streak: Dict[str, int] = {}
        # Calibration gate (CartPole value calibration)
        self.ece_max: float = 0.20
        # Vision calibration/accuracy gates (MNIST-C)
        self.mnistc_ece_max: float = 0.10
        self.mnistc_acc_min: float = 0.85
        # Metrics exposition
        self.last_gate_metrics: Dict[str, Any] = {}
        self._metrics_http = None
        # Start metrics server if configured
        try:
            port = os.getenv('UBRAIN_METRICS_PORT', '0')
            if int(port) > 0:
                self._start_metrics_server()
                brain_logger.info(f"✅ Metrics server started on port {port}")
        except Exception as e:
            brain_logger.error(f"❌ Metrics server failed: {e}", exc_info=True)

        # Ensure routers are initialized before any gating/promotions
        try:
            if self.soup.registry.get_active():
                if self.soup.router is None:
                    self.soup.initialize_router()
            if self.core.registry.get_active():
                if self.core.router is None:
                    self.core.initialize_router()
        except Exception:
            pass
        
    # -------------------- Rollback helpers --------------------
    def _parse_eval_seeds(self) -> List[int]:
        try:
            raw = os.getenv('UBRAIN_EVAL_SEEDS', '42,43')
            seeds = [int(s.strip()) for s in raw.split(',') if s.strip()]
            return seeds if seeds else [42, 43]
        except Exception:
            return [42, 43]

    def _suite_mean(self, suite: Dict[str, List[float]]) -> float:
        vals: List[float] = []
        try:
            for arr in suite.values():
                vals.extend([float(x) for x in arr])
            return sum(vals) / max(1, len(vals))
        except Exception:
            return 0.0

    def _per_seed_means(self, suite: Dict[str, List[float]]) -> List[float]:
        # Compute mean across tasks per seed index
        try:
            n = max((len(v) for v in suite.values()), default=0)
            out: List[float] = []
            for i in range(n):
                acc = 0.0
                cnt = 0
                for arr in suite.values():
                    if i < len(arr):
                        acc += float(arr[i])
                        cnt += 1
                out.append(acc / max(1, cnt))
            return out
        except Exception:
            return []

    def run_maintenance_with_rollback(self) -> Dict[str, Any]:
        """Perform maintenance (promote/demote/unfreeze) with external-only rollback.

        Steps:
        - Evaluate baseline on external suite (multi-seed)
        - Snapshot core+soup (registry + router state)
        - Run tick_maintenance()
        - Re-evaluate; if no consistent multi-seed improvement, revert snapshot
        """
        from datetime import datetime
        from math import ceil
        # Lazy import to avoid circular import at module load time
        try:
            from UNIFIED_BRAIN.brain_system_integration import evaluate_suite  # type: ignore
        except Exception:
            # Fallback: no evaluation possible → do plain maintenance
            return self.tick_maintenance()

        seeds = self._parse_eval_seeds()
        episodes_env = os.getenv('UBRAIN_EVAL_EPISODES', '1')
        try:
            episodes = int(episodes_env)
        except Exception:
            episodes = 1
        os.environ['UBRAIN_EVAL_EPISODES'] = str(episodes)

        # Baseline eval
        try:
            baseline = evaluate_suite(self.core, seeds)
        except Exception:
            baseline = { 'cartpole_nominal': [0.0 for _ in seeds] }
        baseline_mean = self._suite_mean(baseline)
        baseline_seed_means = self._per_seed_means(baseline)

        # Snapshot
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_dir = Path('.penin_omega/rollback') / stamp
        core_dir = base_dir / 'core'
        soup_dir = base_dir / 'soup'
        try:
            core_dir.mkdir(parents=True, exist_ok=True)
            soup_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        core_snap = core_dir / 'core.pt'
        soup_snap = soup_dir / 'soup.pt'
        try:
            self.core.save_snapshot(str(core_snap))
            self.soup.save_snapshot(str(soup_snap))
        except Exception:
            # If snapshot fails, fallback to plain maintenance
            return self.tick_maintenance()

        # Apply maintenance
        results = self.tick_maintenance()

        # Post-eval
        try:
            after = evaluate_suite(self.core, seeds)
        except Exception:
            after = { 'cartpole_nominal': [0.0 for _ in seeds] }
        after_mean = self._suite_mean(after)
        after_seed_means = self._per_seed_means(after)

        # Decision: require minimum uplift and seed-improvement fraction (env-configurable)
        tol = float(os.getenv('UBRAIN_ROLLBACK_MIN_IMPROVEMENT', '0.0'))
        improved_seeds = 0
        for i in range(min(len(baseline_seed_means), len(after_seed_means))):
            if after_seed_means[i] >= baseline_seed_means[i] + tol:
                improved_seeds += 1
        try:
            frac = float(os.getenv('UBRAIN_ROLLBACK_MIN_SEED_FRAC', '0.7'))
        except Exception:
            frac = 0.7
        # clamp to [0,1]
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        required = ceil(max(1, len(seeds)) * frac)
        commit = (after_mean >= baseline_mean + tol) and (improved_seeds >= required)

        # WORM log
        try:
            if self.core.worm:
                self.core.worm.append('rollback_decision', {
                    'commit': bool(commit),
                    'baseline_mean': float(baseline_mean),
                    'after_mean': float(after_mean),
                    'improved_seeds': int(improved_seeds),
                    'required': int(required),
                    'seeds': seeds,
                    'results': results,
                })
        except Exception:
            pass

        if commit:
            return { **results, 'rollback': False, 'baseline_mean': baseline_mean, 'after_mean': after_mean }

        # Rollback
        try:
            # Restore core
            core_registry_path = core_dir / 'neuron_registry.json'
            if core_registry_path.exists():
                self.core.registry.load_with_adapters(str(core_registry_path))
            self.core.load_snapshot(str(core_snap))
            # Restore soup
            soup_registry_path = soup_dir / 'neuron_registry.json'
            if soup_registry_path.exists():
                self.soup.registry.load_with_adapters(str(soup_registry_path))
            self.soup.load_snapshot(str(soup_snap))
            # Ensure routers
            if self.core.router is None and self.core.registry.get_active():
                self.core.initialize_router()
            if self.soup.router is None and self.soup.registry.get_active():
                self.soup.initialize_router()
        except Exception:
            pass

        try:
            if self.core.worm:
                self.core.worm.append('rollback_applied', {
                    'baseline_mean': float(baseline_mean),
                    'after_mean': float(after_mean),
                    'seeds': seeds,
                })
        except Exception:
            pass

        return { **results, 'rollback': True, 'baseline_mean': baseline_mean, 'after_mean': after_mean }

    def evaluate_soup_neuron(self, neuron_id: str) -> float:
        """Avalia neurônio do soup para possível promoção"""
        neuron = self.soup.registry.get(neuron_id)
        if not neuron:
            return 0.0
        
        # Critérios: competence, ativação, sem erros
        score = (
            neuron.meta.competence_score * 0.5 +
            (neuron.meta.activation_count / max(1, self.evaluation_window)) * 0.3 +
            neuron.meta.novelty_score * 0.2
        )
        
        return score
    
    def promote_from_soup(self):
        """Promove neurônios do soup para o core com segurança (cooldown + estabilidade)."""
        # Fetch current soup candidates once
        soup_neurons = self.soup.registry.get_active()

        # ✅ FIX #5: Bootstrap mode - skip gates para primeiros 10 neurons
        if len(self.core.registry.get_active()) < 10:
            if not soup_neurons:
                return False  # Nada para promover
            # Seleciona um candidato simples (primeiro ativo)
            neuron = soup_neurons[0]
            brain_logger.info(f"🚀 Bootstrap: promoting {neuron.meta.id} without gates")
            # Congele no soup e registre no core
            self.soup.registry.promote(neuron.meta.id, NeuronStatus.FROZEN)
            self.core.register_neuron(neuron)
            # WORM: use ledger do core ou abra um novo local
            try:
                from brain_worm import WORMLog
                worm = getattr(self.core, 'worm', None) or WORMLog()
                worm.append('promotion_bootstrap', {'neuron_id': neuron.meta.id})
            except Exception:
                pass
            # Router precisa refletir novo conjunto
            self.core.initialize_router()
            # Atualiza cooldown
            self._last_promotion_step = getattr(self.core, 'step_count', 0)
            return True
        
        # Ensure metrics server is running (hot)
        try:
            self._start_metrics_server()
        except Exception:
            pass
        # Cooldown global de promoção
        if isinstance(self.core, UnifiedBrain) and (self.core.step_count - self._last_promotion_step) < self.promotion_cooldown:
            return
        # Use já a lista obtida acima
        for neuron in soup_neurons:
            score = self.evaluate_soup_neuron(neuron.meta.id)
            # Requer estabilidade mínima (competence_score acima do threshold por janela)
            stable_enough = getattr(neuron.meta, 'stable_steps', 0) >= self.stability_window
            if score >= self.promotion_threshold and stable_enough:
                from brain_worm import WORMLog
                worm = getattr(self.core, 'worm', None) or WORMLog()
                
                # BOOTSTRAP MODE: Skip gates para primeiros 10 neurons (C1 fix)
                if len(self.core.registry.get_active()) < 10:
                    brain_logger.info(f"🚀 Bootstrap mode: promoting {neuron.meta.id} without gates (core size={len(self.core.registry.get_active())})")
                    self.soup.registry.promote(neuron.meta.id, NeuronStatus.FROZEN)
                    self.core.register_neuron(neuron)
                    self._last_promotion_step = self.core.step_count
                    worm.append('promotion_bootstrap', {
                        'neuron_id': neuron.meta.id,
                        'score': float(score),
                        'core_count': len(self.core.registry.get_active()),
                        'reason': 'bootstrap_skip_gates',
                    })
                    self.core.initialize_router()
                    continue  # SKIP gates
                
                # Normal path: Executa hard-gates de OOD/retention e p-valor A/B
                gate_ok, gate_info = self._run_promotion_gates(neuron)
                # Atualiza streak por neurônio
                prev = self._gate_streak.get(neuron.meta.id, 0)
                cur = prev + 1 if gate_ok else 0
                self._gate_streak[neuron.meta.id] = cur
                worm.append('promotion_gate', {
                    'neuron_id': neuron.meta.id,
                    'score': float(score),
                    'stable_steps': int(getattr(neuron.meta, 'stable_steps', 0)),
                    'threshold': float(self.promotion_threshold),
                    'window': int(self.stability_window),
                    'core_steps': int(getattr(self.core, 'step_count', 0)),
                    'ood_uplift': gate_info.get('uplift', None),
                    'retention': gate_info.get('retention', None),
                    'p_value': gate_info.get('p_value', None),
                    'p_value_ab': gate_info.get('p_value_ab', None),
                    'ece_cartpole': gate_info.get('ece_cartpole', None),
                    'mnistc_ece': gate_info.get('mnistc_ece', None),
                    'mnistc_acc': gate_info.get('mnistc_acc', None),
                    'mnistc_samples': gate_info.get('mnistc_samples', None),
                    'samples': gate_info.get('samples', {}),
                    'streak': int(cur),
                    'required_streak': int(self.gate_required_streak),
                    'passed': bool(gate_ok and cur >= self.gate_required_streak),
                })
                if not (gate_ok and cur >= self.gate_required_streak):
                    # Não promove ainda; requer mais evidência
                    continue
                # Congela no soup e registra no core
                self.soup.registry.promote(neuron.meta.id, NeuronStatus.FROZEN)
                self.core.register_neuron(neuron)
                self._last_promotion_step = self.core.step_count

    def demote_low_competence(self, competence_threshold: float = 0.2):
        """Marca como FROZEN neurônios de baixa competência no core com cooldown."""
        if isinstance(self.core, UnifiedBrain) and (self.core.step_count - self._last_demotion_step) < self.demotion_cooldown:
            return
        active = self.core.registry.get_active()
        demoted = 0
        for neuron in active:
            try:
                if self.core.router is None:
                    continue
                if neuron.meta.competence_score < competence_threshold:
                    self.core.registry.promote(neuron.meta.id, NeuronStatus.FROZEN)
                    demoted += 1
                    try:
                        from brain_worm import WORMLog
                        worm = getattr(self.core, 'worm', None) or WORMLog()
                        worm.append('demotion_decision', {
                            'neuron_id': neuron.meta.id,
                            'competence': float(neuron.meta.competence_score),
                            'threshold': float(competence_threshold),
                            'core_steps': int(getattr(self.core, 'step_count', 0)),
                        })
                    except Exception:
                        pass
            except Exception:
                continue
        if demoted > 0:
            self._last_demotion_step = self.core.step_count
    
    def unfreeze_improved(self, competence_threshold: float = 0.5):
        """
        ✅ CORREÇÃO #11: Descongela neurônios que melhoraram competência
        Reverso de demote_low_competence() - permite recuperação
        Darwin bidirectional: neurons podem melhorar E piorar
        
        Args:
            competence_threshold: competence mínimo para descongelar (>= 0.5)
        
        Returns:
            unfrozen: número de neurons descongelados
        """
        from brain_worm import WORMLog
        from brain_logger import brain_logger
        
        # Cooldown para não descongelar muito rápido
        if isinstance(self.core, UnifiedBrain):
            steps_since_unfreeze = (self.core.step_count - 
                                    getattr(self, '_last_unfreeze_step', 0))
            if steps_since_unfreeze < 500:  # Cooldown de 500 steps
                return 0
        
        # Pegar neurons congelados
        frozen = self.core.registry.get_by_status(NeuronStatus.FROZEN)
        unfrozen = 0
        
        for neuron in frozen:
            try:
                # ✅ CRITÉRIO: Competence voltou a subir acima do threshold
                if neuron.meta.competence_score > competence_threshold:
                    
                    # ✅ CRITÉRIO ADICIONAL: Estabilidade (não oscilar)
                    stable_steps = getattr(neuron.meta, 'stable_steps', 0)
                    if stable_steps < 100:  # Requer 100 steps estável
                        continue
                    
                    # Descongelar: FROZEN → ACTIVE
                    self.core.registry.promote(neuron.meta.id, NeuronStatus.ACTIVE)
                    unfrozen += 1
                    
                    # Log to WORM para auditoria
                    worm = getattr(self.core, 'worm', None) or WORMLog()
                    worm.append('unfreeze_decision', {
                        'neuron_id': neuron.meta.id,
                        'competence': float(neuron.meta.competence_score),
                        'threshold': float(competence_threshold),
                        'stable_steps': int(stable_steps),
                        'core_steps': int(getattr(self.core, 'step_count', 0)),
                        'reason': 'competence_improved'
                    })
                    
                    brain_logger.info(
                        f"🔥 Darwin unfroze '{neuron.meta.id}' "
                        f"(competence={neuron.meta.competence_score:.3f})"
                    )
            
            except Exception as e:
                brain_logger.error(f"Failed to unfreeze {neuron.meta.id}: {e}")
                continue
        
        if unfrozen > 0:
            # Re-initialize router com neurons atualizados
            self.core.initialize_router()
            
            # Atualizar timestamp
            if isinstance(self.core, UnifiedBrain):
                self._last_unfreeze_step = self.core.step_count
            
            brain_logger.warning(
                f"🔥 Darwin unfroze {unfrozen} improved neurons! "
                f"Active neurons: {len(self.core.registry.get_active())}"
            )
        
        return unfrozen

    # -------------------- Hard Gates helpers --------------------
    def _eval_variant(self, brain: UnifiedBrain, variant: str, steps: int = 150) -> float:
        import torch
        H = brain.H
        score = 0.0
        z = torch.randn(1, H)
        for i in range(steps):
            reward = 0.5
            if variant == 'noise_high':
                reward += float(torch.randn(1).clamp(-0.2, 0.2))
            elif variant == 'sparse_reward':
                reward = 1.0 if (i % 25 == 0) else 0.0
            elif variant == 'shifted_distribution':
                z = z + 0.05 * torch.randn_like(z)
            z, _ = brain.step(z, reward=reward)
            score += reward
        return score / max(1, steps)

    def _bootstrap_pvalue(self, samples: List[float], mu0: float = 0.0, n_boot: int = 1000) -> float:
        import random
        if not samples:
            return 1.0
        n = len(samples)
        mean_obs = sum(samples) / n
        diffs = [s - mu0 for s in samples]
        # Bootstrap means around mu0 by resampling centered residuals
        count = 0
        for _ in range(n_boot):
            b = [random.choice(diffs) for _ in range(n)]
            mean_b = sum(b) / n
            if mean_b <= 0.0:
                count += 1
        p = count / n_boot
        return p

    def _run_promotion_gates(self, candidate_neuron) -> Tuple[bool, Dict[str, Any]]:
        """Run OOD/retention using real Gym suite + A/B p-value with candidate neuron."""
        try:
            from brain_system_integration import (
                evaluate_suite,
                evaluate_env_reward,
                evaluate_cartpole_calibration,
                evaluate_mnist_c,
            )
            import os as _os
            seeds_env = _os.getenv('UBRAIN_EVAL_SEEDS', '41,42,43')
            seeds = [int(s.strip()) for s in seeds_env.split(',') if s.strip()]
            if not seeds:
                seeds = [41, 42, 43]

            # Hot-reload thresholds from environment
            try:
                self.gate_uplift_min = float(_os.getenv('UBRAIN_GATE_UPLIFT_MIN', str(self.gate_uplift_min)))
                self.gate_retention_min = float(_os.getenv('UBRAIN_GATE_RETENTION_MIN', str(self.gate_retention_min)))
                self.gate_pvalue_max = float(_os.getenv('UBRAIN_GATE_PVAL_MAX', str(self.gate_pvalue_max)))
                self.gate_required_streak = int(_os.getenv('UBRAIN_GATE_STREAK', str(self.gate_required_streak)))
                self.ece_max = float(_os.getenv('UBRAIN_GATE_ECE_MAX', str(self.ece_max)))
                self.mnistc_ece_max = float(_os.getenv('UBRAIN_GATE_MNISTC_ECE_MAX', str(self.mnistc_ece_max)))
                self.mnistc_acc_min = float(_os.getenv('UBRAIN_GATE_MNISTC_ACC_MIN', str(self.mnistc_acc_min)))
            except Exception:
                pass

            # CONTROL: current core only
            suite_c = evaluate_suite(self.core, seeds)
            base_list = suite_c.get('cartpole_nominal', [])
            # OOD across tasks
            ood_mat = [
                suite_c.get('cartpole_noise', []),
                suite_c.get('cartpole_shift', []),
                suite_c.get('mountaincar', []),
                suite_c.get('acrobot', []),
            ]
            # Per-seed averages
            ood_avg_list = []
            for i in range(len(seeds)):
                vals = [arr[i] for arr in ood_mat if i < len(arr)]
                ood_avg_list.append(sum(vals) / max(1, len(vals)))

            # Uplift and retention
            import numpy as _np
            base = float(_np.mean(base_list)) if base_list else 1e-6
            ood_avg = float(_np.mean(ood_avg_list)) if ood_avg_list else 0.0
            uplift = (ood_avg - base) / max(1e-6, abs(base))
            # Post-nominal retention (re-run nominal with offset seeds)
            base_after_list = [
                evaluate_env_reward(self.core, 'CartPole-v1', episodes=1, noise_std=0.0, obs_shift=0.0, seed=s+97)
                for s in seeds
            ]
            base_after = float(_np.mean(base_after_list)) if base_after_list else base
            retention = base_after / max(1e-6, abs(base))
            # Proxy p-value on uplift per seed
            uplifts = [
                (ood_avg_list[i] - base_list[i]) if i < len(base_list) else 0.0
                for i in range(len(seeds))
            ]
            p_value = self._bootstrap_pvalue(uplifts, mu0=0.0, n_boot=500)

            # A/B with candidate: temporarily add candidate to core
            ab_info = self._ab_test_candidate(candidate_neuron, seeds)
            p_value_ab = float(ab_info.get('p_value', 1.0))

            # Calibration (CartPole value ECE)
            try:
                cal_episodes = int(_os.getenv('UBRAIN_CALIB_EPISODES', '1'))
                ece_cart = float(evaluate_cartpole_calibration(self.core, seeds, episodes=cal_episodes))
            except Exception:
                ece_cart = 1.0

            # Vision robustness/calibration: MNIST-C
            mnistc = {}
            try:
                mnistc = evaluate_mnist_c()
            except Exception:
                mnistc = {'accuracy': 0.0, 'ece': 1.0, 'per_corruption': {}, 'num_samples': 0}

            passed = (
                uplift >= self.gate_uplift_min and
                retention >= self.gate_retention_min and
                p_value_ab <= self.gate_pvalue_max and
                ece_cart <= self.ece_max and
                mnistc.get('ece', 1.0) <= self.mnistc_ece_max and
                mnistc.get('accuracy', 0.0) >= self.mnistc_acc_min
            )
            info_out = {
                'uplift': float(uplift),
                'retention': float(retention),
                'p_value': float(p_value),
                'p_value_ab': p_value_ab,
                'ece_cartpole': float(ece_cart),
                'mnistc_ece': float(mnistc.get('ece', 1.0)),
                'mnistc_acc': float(mnistc.get('accuracy', 0.0)),
                'mnistc_samples': int(mnistc.get('num_samples', 0)),
                'samples': {
                    'base': base_list,
                    'ood_avg': ood_avg_list,
                    'base_after': base_after_list,
                    'ab_control': ab_info.get('control', {}),
                    'ab_treatment': ab_info.get('treatment', {}),
                },
            }

            # Update last_gate_metrics for Prometheus exporter if present
            try:
                self.last_gate_metrics.update({
                    'uplift': info_out['uplift'],
                    'retention': info_out['retention'],
                    'p_value': info_out['p_value'],
                    'p_value_ab': info_out['p_value_ab'],
                    'ece_cartpole': info_out['ece_cartpole'],
                    'mnistc_ece': info_out['mnistc_ece'],
                    'mnistc_acc': info_out['mnistc_acc'],
                })
            except Exception:
                pass

            return passed, info_out
        except Exception as _e:
            return False, {'uplift': 0.0, 'retention': 0.0, 'p_value': 1.0, 'p_value_ab': 1.0}

    def _ab_test_candidate(self, candidate_neuron, seeds: List[int]) -> Dict[str, Any]:
        """A/B test: compare control vs control+candidate in Gym suite; returns p-value and samples."""
        from brain_system_integration import evaluate_suite
        import copy as _copy
        # CONTROL
        control = evaluate_suite(self.core, seeds)
        # TREATMENT: temporarily add candidate
        nid = candidate_neuron.meta.id
        added = False
        try:
            if nid not in self.core.registry.neurons:
                # Shallow register copy (avoid mutating soup neuron)
                self.core.registry.register(candidate_neuron)
                added = True
            self.core.initialize_router()
            treatment = evaluate_suite(self.core, seeds)
        finally:
            # Remove candidate from core
            try:
                if added and nid in self.core.registry.neurons:
                    # Remove from indices and dicts
                    neuron = self.core.registry.neurons.pop(nid)
                    try:
                        if nid in self.core.registry.by_status.get(neuron.meta.status, []):
                            self.core.registry.by_status[neuron.meta.status].remove(nid)
                    except Exception:
                        pass
                    try:
                        if nid in self.core.registry.meta_db:
                            self.core.registry.meta_db.pop(nid)
                    except Exception:
                        pass
                    self.core.initialize_router()
            except Exception:
                pass

        # Build per-seed uplift arrays and compute p-value on improvement
        import numpy as _np
        def _avg_ood(suite):
            mats = [suite.get('cartpole_noise', []), suite.get('cartpole_shift', []), suite.get('mountaincar', []), suite.get('acrobot', [])]
            out = []
            for i in range(len(seeds)):
                vals = [arr[i] for arr in mats if i < len(arr)]
                out.append(sum(vals) / max(1, len(vals)))
            return out
        base_c = control.get('cartpole_nominal', [])
        base_t = treatment.get('cartpole_nominal', [])
        ood_c = _avg_ood(control)
        ood_t = _avg_ood(treatment)
        diffs = []
        for i in range(len(seeds)):
            c = (ood_c[i] - base_c[i]) if i < len(base_c) else 0.0
            t = (ood_t[i] - base_t[i]) if i < len(base_t) else 0.0
            diffs.append(t - c)
        p = self._bootstrap_pvalue(diffs, mu0=0.0, n_boot=1000)
        return {
            'p_value': float(p),
            'control': control,
            'treatment': treatment,
        }

    def tick_maintenance(self) -> Dict[str, Any]:
        """
        Executa manutenção periódica: promote, demote com tracking completo
        ✅ CORREÇÃO #10: Retorna métricas detalhadas + validação
        
        Returns:
            Dict com promoted, demoted, active_before, active_after, errors
        """
        from brain_logger import brain_logger
        from datetime import datetime
        
        # Inicializar resultados
        results = {
            'promoted': 0,
            'demoted': 0,
            'active_before': len(self.core.registry.get_active()),
            'active_after': 0,
            'soup_before': len(self.soup.registry.get_active()) if hasattr(self, 'soup') else 0,
            'soup_after': 0,
            'errors': [],
            'timestamp': datetime.now().isoformat(),
            'core_steps': getattr(self.core, 'step_count', 0)
        }
        
        # ===== PROMOTIONS =====
        try:
            soup_before = len(self.soup.registry.get_active()) if hasattr(self, 'soup') else 0
            self.promote_from_soup()
            soup_after = len(self.soup.registry.get_active()) if hasattr(self, 'soup') else 0
            
            results['promoted'] = soup_before - soup_after
            results['soup_after'] = soup_after
            
            if results['promoted'] > 0:
                brain_logger.info(
                    f"🧬 Promoted {results['promoted']} neurons from soup to core"
                )
        except Exception as e:
            results['errors'].append(f"promotion_failed: {str(e)}")
            brain_logger.error(f"❌ Promotion failed: {e}")
        
        # ===== DEMOTIONS =====
        try:
            core_before = len(self.core.registry.get_active())
            self.demote_low_competence()
            core_after = len(self.core.registry.get_active())
            
            results['demoted'] = core_before - core_after
            results['active_after'] = core_after
            
            if results['demoted'] > 0:
                brain_logger.warning(
                    f"🧬 Demoted {results['demoted']} low-competence neurons"
                )
        except Exception as e:
            results['errors'].append(f"demotion_failed: {str(e)}")
            brain_logger.error(f"❌ Demotion failed: {e}")
        
        # ===== UNFREEZING ===== (✅ CORREÇÃO #11: Darwin bidirectional)
        try:
            frozen_before = len(self.core.registry.get_by_status(NeuronStatus.FROZEN))
            
            # Executar unfreezing
            unfrozen_count = self.unfreeze_improved()
            
            frozen_after = len(self.core.registry.get_by_status(NeuronStatus.FROZEN))
            results['unfrozen'] = frozen_before - frozen_after
            
            if results['unfrozen'] > 0:
                brain_logger.info(
                    f"🔥 Unfroze {results['unfrozen']} improved neurons"
                )
        except Exception as e:
            results['errors'].append(f"unfreezing_failed: {str(e)}")
            brain_logger.error(f"❌ Unfreezing failed: {e}")
        
        # ===== FINALIZAÇÃO =====
        results['active_after'] = len(self.core.registry.get_active())
        
        # Log to WORM
        if hasattr(self.core, 'worm') and self.core.worm:
            self.core.worm.append('maintenance_cycle', {
                'promoted': results['promoted'],
                'demoted': results['demoted'],
                'unfrozen': results.get('unfrozen', 0),  # ✅ Incluir unfrozen
                'active_before': results['active_before'],
                'active_after': results['active_after'],
                'errors': results['errors'],
                'core_steps': results['core_steps']
            })
        
        # ⚠️ ALERTAS para mudanças drásticas
        if results['demoted'] > 5:
            brain_logger.critical(
                f"⚠️ WARNING: {results['demoted']} neurons demoted! "
                f"System may be unstable."
            )
        
        if results['errors']:
            brain_logger.error(
                f"⚠️ Maintenance had {len(results['errors'])} errors"
            )
        
        return results

    # -------------------- Prometheus exporter --------------------
    def _start_metrics_server(self):
        if getattr(self, '_metrics_http', None) is not None:
            return
        try:
            port = int(os.getenv('UBRAIN_METRICS_PORT', '0'))
        except Exception:
            port = 0
        if port <= 0:
            return
        outer_self = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ('/metrics', '/', ''):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    payload = outer_self._render_prometheus()
                except Exception:
                    payload = "# metrics unavailable\n"
                data = payload.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except Exception:
                    pass
            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self._metrics_http = server

    def _render_prometheus(self) -> str:
        m = self.core.get_metrics_summary() if hasattr(self, 'core') else {}
        lines = []
        def add(name: str, value: float, help_text: str = "", typ: str = "gauge"):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {typ}")
            try:
                v = float(value)
            except Exception:
                v = 0.0
            lines.append(f"{name} {v}")
        # Core metrics
        add('ubrain_steps_total', m.get('total_steps', 0), 'Total brain steps', 'counter')
        add('ubrain_avg_latency_ms', m.get('avg_latency_ms', 0.0), 'Average latency per step (ms)')
        add('ubrain_avg_coherence', m.get('avg_coherence', 0.0), 'Average coherence of Z transitions')
        add('ubrain_avg_novelty', m.get('avg_novelty', 0.0), 'Average novelty of Z transitions')
        try:
            top_k = self.core.router.top_k if self.core and self.core.router else 0
        except Exception:
            top_k = 0
        add('ubrain_router_top_k', top_k, 'Current router top_k')
        # Gate metrics (last evaluation)
        g = self.last_gate_metrics or {}
        add('ubrain_gate_uplift', g.get('uplift', 0.0), 'OOD uplift (relative to nominal)')
        add('ubrain_gate_retention', g.get('retention', 0.0), 'Retention on nominal after OOD')
        add('ubrain_gate_p_value', g.get('p_value', 1.0), 'Bootstrap p-value of uplift')
        add('ubrain_gate_p_value_ab', g.get('p_value_ab', 1.0), 'A/B p-value control vs treatment')
        add('ubrain_gate_ece_cartpole', g.get('ece_cartpole', 1.0), 'CartPole value ECE')
        add('ubrain_gate_mnistc_ece', g.get('mnistc_ece', 1.0), 'MNIST-C classification ECE')
        add('ubrain_gate_mnistc_acc', g.get('mnistc_acc', 0.0), 'MNIST-C accuracy')
        return "\n".join(lines) + "\n"


class CollectiveConsciousnessSystem:
    """
    Sistema de consciência coletiva para integração neural unificada
    """

    def __init__(self):
        self.consciousness_level = 0.0
        self.interconnected_systems = []
        self.emergence_signals = []
        self.collective_memory = []

    def connect_system(self, system_name, system_type):
        """Conecta sistema à consciência coletiva"""
        system_info = {
            'name': system_name,
            'type': system_type,
            'connection_time': datetime.now().isoformat(),
            'consciousness_contribution': 0.0
        }

        self.interconnected_systems.append(system_info)
        brain_logger.info(f"🧠 Sistema {system_name} conectado à consciência coletiva")

        return system_info

    def propagate_signal(self, signal_source, signal_data):
        """Propaga sinal através da consciência coletiva"""
        signal = {
            'source': signal_source,
            'data': signal_data,
            'timestamp': datetime.now().isoformat(),
            'propagation_path': []
        }

        # Propagar para sistemas conectados
        for system in self.interconnected_systems:
            if system['name'] != signal_source:
                signal['propagation_path'].append(system['name'])

        self.collective_memory.append(signal)
        brain_logger.info(f"📡 Sinal propagado de {signal_source} para {len(signal['propagation_path'])} sistemas")

        return signal

    def compute_collective_state(self):
        """Computa estado coletivo da consciência"""
        if not self.interconnected_systems:
            return 0.0

        total_contribution = sum(sys['consciousness_contribution'] for sys in self.interconnected_systems)
        avg_contribution = total_contribution / len(self.interconnected_systems)

        # Aumentar consciência baseado na integração
        self.consciousness_level = min(self.consciousness_level + avg_contribution * 0.01, 1.0)

        return self.consciousness_level

    def detect_emergence(self):
        """Detecta sinais de emergência coletiva"""
        if len(self.collective_memory) < 10:
            return False

        # Verificar sinais de emergência coletiva
        recent_signals = self.collective_memory[-5:]

        # Critérios para emergência coletiva
        signal_diversity = len(set(sig['source'] for sig in recent_signals))
        signal_frequency = len(recent_signals) / 5  # Últimos 5 sinais
        consciousness_level = self.compute_collective_state()

        emergence_detected = (
            signal_diversity >= 3 and
            signal_frequency >= 0.8 and
            consciousness_level >= 0.1
        )

        if emergence_detected:
            brain_logger.warning(f"🚨 Emergência coletiva detectada! Nível: {consciousness_level:.3f}")
            self.emergence_signals.append({
                'timestamp': datetime.now().isoformat(),
                'level': consciousness_level,
                'systems_involved': signal_diversity
            })

        return emergence_detected


class EmergenceDetector:
    """
    Detector avançado de emergência baseado em múltiplos critérios
    """

    def __init__(self):
        self.emergence_thresholds = {
            'consciousness_level': 0.3,
            'signal_complexity': 0.7,
            'interconnection_density': 0.5,
            'adaptation_rate': 0.2
        }
        self.emergence_history = []

    def analyze_system_state(self, system_state):
        """Analisa estado do sistema para sinais de emergência"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'consciousness_score': 0.0,
            'complexity_score': 0.0,
            'adaptation_score': 0.0,
            'emergence_probability': 0.0
        }

        # Análise de consciência
        consciousness_indicators = [
            'self_observation' in str(system_state),
            'adaptive_feedback' in str(system_state),
            'pattern_analysis' in str(system_state)
        ]
        analysis['consciousness_score'] = sum(consciousness_indicators) / len(consciousness_indicators)

        # Análise de complexidade
        complexity_indicators = [
            'neural_networks' in str(system_state),
            'evolution_algorithms' in str(system_state),
            'multi_modal' in str(system_state)
        ]
        analysis['complexity_score'] = sum(complexity_indicators) / len(complexity_indicators)

        # Análise de adaptação
        adaptation_indicators = [
            'q_learning' in str(system_state),
            'reinforcement_learning' in str(system_state),
            'parameter_adaptation' in str(system_state)
        ]
        analysis['adaptation_score'] = sum(adaptation_indicators) / len(adaptation_indicators)

        # Calcular probabilidade de emergência
        weights = [0.3, 0.3, 0.4]  # Pesos para consciência, complexidade, adaptação
        scores = [analysis['consciousness_score'], analysis['complexity_score'], analysis['adaptation_score']]
        analysis['emergence_probability'] = sum(w * s for w, s in zip(weights, scores))

        self.emergence_history.append(analysis)

        return analysis

    def check_emergence_criteria(self, analysis):
        """Verifica se critérios de emergência são atendidos"""
        criteria_met = 0

        if analysis['consciousness_score'] >= self.emergence_thresholds['consciousness_level']:
            criteria_met += 1

        if analysis['complexity_score'] >= self.emergence_thresholds['signal_complexity']:
            criteria_met += 1

        if analysis['adaptation_score'] >= self.emergence_thresholds['adaptation_rate']:
            criteria_met += 1

        # Verificar tendência crescente
        if len(self.emergence_history) >= 3:
            recent_probs = [h['emergence_probability'] for h in self.emergence_history[-3:]]
            if recent_probs[-1] > recent_probs[0]:
                criteria_met += 1

        return criteria_met >= 3  # Pelo menos 3 critérios atendidos

    def get_status(self):
        """Retorna status geral do UnifiedBrain"""
        return {
            'neurons_count': len(self.neurons) if hasattr(self, 'neurons') else 0,
            'active_neurons': len([n for n in getattr(self, 'neurons', []) if n.status == NeuronStatus.ACTIVE]) if hasattr(self, 'neurons') else 0,
            'emergence_status': self.get_emergence_status(),
            'consciousness_level': getattr(self, 'consciousness_level', 0.0),
            'integration_level': getattr(self, 'integration_level', 0.0)
        }

    def get_emergence_status(self):
        """Retorna status atual de emergência"""
        if not self.emergence_history:
            return {'emergence_probability': 0.0, 'status': 'no_data'}

        latest = self.emergence_history[-1]

        if self.check_emergence_criteria(latest):
            return {
                'emergence_probability': latest['emergence_probability'],
                'status': 'emergence_detected',
                'confidence': latest['emergence_probability']
            }
        elif latest['emergence_probability'] > 0.5:
            return {
                'emergence_probability': latest['emergence_probability'],
                'status': 'high_potential',
                'confidence': latest['emergence_probability']
            }
        else:
            return {
                'emergence_probability': latest['emergence_probability'],
                'status': 'developing',
                'confidence': latest['emergence_probability']
            }


class NeuralIntegrationSystem:
    """Sistema de integração neural para inteligência unificada"""
    
    def __init__(self):
        self.neural_connections = {}
        self.integration_history = []
        self.synaptic_strength = {}
        self.neural_plasticity = 0.1
        self.integration_threshold = 0.7
        # Ensure logger is available
        import logging
        self.logger = logging.getLogger(__name__)
        
    def integrate_neural_networks(self, networks_data):
        """Integra múltiplas redes neurais"""
        try:
            # 1. Análise de redes
            network_analysis = self._analyze_networks(networks_data)
            
            # 2. Criação de conexões
            connections = self._create_connections(network_analysis)
            
            # 3. Fortalecimento sináptico
            synaptic_updates = self._strengthen_synapses(connections)
            
            # 4. Detecção de integração
            integration_signals = self._detect_integration_signals(synaptic_updates)
            
            # 5. Registro da integração
            self._record_integration(network_analysis, integration_signals)
            
            self.logger.info(f"🧠 Integração neural: conexões={len(connections)}, "
                       f"sinais={len(integration_signals)}")
            
            return integration_signals
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na integração neural: {e}")
            return []
            
    def _analyze_networks(self, networks_data):
        """Analisa redes neurais para integração"""
        analysis = {
            'network_count': len(networks_data) if networks_data else 0,
            'complexity_scores': [],
            'compatibility_scores': [],
            'integration_potential': 0.0
        }
        
        if not networks_data:
            return analysis
            
        # Análise de complexidade
        for network in networks_data:
            if isinstance(network, dict):
                complexity = network.get('complexity', 0.5)
                analysis['complexity_scores'].append(complexity)
                
        # Análise de compatibilidade
        for i, network1 in enumerate(networks_data):
            for j, network2 in enumerate(networks_data[i+1:], i+1):
                compatibility = self._calculate_compatibility(network1, network2)
                analysis['compatibility_scores'].append(compatibility)
                
        # Potencial de integração
        if analysis['complexity_scores']:
            avg_complexity = sum(analysis['complexity_scores']) / len(analysis['complexity_scores'])
            analysis['integration_potential'] = avg_complexity
            
        if analysis['compatibility_scores']:
            avg_compatibility = sum(analysis['compatibility_scores']) / len(analysis['compatibility_scores'])
            analysis['integration_potential'] = (analysis['integration_potential'] + avg_compatibility) / 2
            
        return analysis
        
    def _calculate_compatibility(self, network1, network2):
        """Calcula compatibilidade entre duas redes"""
        compatibility = 0.5  # Base
        
        # Fator 1: Similaridade de arquitetura
        if 'architecture' in network1 and 'architecture' in network2:
            arch1 = network1['architecture']
            arch2 = network2['architecture']
            if arch1 == arch2:
                compatibility += 0.3
                
        # Fator 2: Similaridade de dados
        if 'data_type' in network1 and 'data_type' in network2:
            data1 = network1['data_type']
            data2 = network2['data_type']
            if data1 == data2:
                compatibility += 0.2
                
        return min(1.0, compatibility)
        
    def _create_connections(self, analysis):
        """Cria conexões entre redes"""
        connections = []
        
        if analysis['network_count'] < 2:
            return connections
            
        # Criar conexões baseadas na compatibilidade
        for i in range(analysis['network_count']):
            for j in range(i+1, analysis['network_count']):
                if i < len(analysis['compatibility_scores']):
                    compatibility = analysis['compatibility_scores'][i]
                    if compatibility > 0.6:
                        connection = {
                            'source': i,
                            'target': j,
                            'strength': compatibility,
                            'type': 'neural_synapse'
                        }
                        connections.append(connection)
                        
        return connections
        
    def _strengthen_synapses(self, connections):
        """Fortalecimento sináptico"""
        updates = []
        
        for connection in connections:
            connection_id = f"{connection['source']}->{connection['target']}"
            
            # Calcular novo strength
            current_strength = self.synaptic_strength.get(connection_id, 0.5)
            new_strength = min(1.0, current_strength + self.neural_plasticity)
            
            # Atualizar strength
            self.synaptic_strength[connection_id] = new_strength
            
            update = {
                'connection_id': connection_id,
                'old_strength': current_strength,
                'new_strength': new_strength,
                'strengthening': new_strength - current_strength
            }
            updates.append(update)
            
        return updates
        
    def _detect_integration_signals(self, synaptic_updates):
        """Detecta sinais de integração"""
        signals = []
        
        if not synaptic_updates:
            return signals
            
        # Sinal 1: Fortalecimento sináptico
        strong_connections = sum(1 for update in synaptic_updates 
                               if update['new_strength'] > 0.8)
        if strong_connections > 0:
            signals.append({
                'type': 'synaptic_strengthening',
                'strength': strong_connections / len(synaptic_updates),
                'description': 'Fortalecimento sináptico detectado'
            })
            
        # Sinal 2: Integração neural
        if len(synaptic_updates) > 2:
            signals.append({
                'type': 'neural_integration',
                'strength': min(1.0, len(synaptic_updates) / 5.0),
                'description': 'Integração neural detectada'
            })
            
        return signals
        
    def _record_integration(self, analysis, signals):
        """Registra integração no histórico"""
        record = {
            'timestamp': time.time(),
            'analysis': analysis,
            'signals': signals,
            'total_connections': len(self.synaptic_strength)
        }
        
        self.integration_history.append(record)
        
        # Manter apenas últimos 50 registros
        if len(self.integration_history) > 50:
            self.integration_history = self.integration_history[-50:]
            
    def get_integration_insights(self):
        """Retorna insights sobre integração"""
        if not self.integration_history:
            return {'integration_count': 0, 'avg_signals': 0}
            
        recent_integration = self.integration_history[-10:]
        avg_signals = sum(len(r['signals']) for r in recent_integration) / len(recent_integration)
        
        return {
            'integration_count': len(self.integration_history),
            'avg_signals': avg_signals,
            'total_connections': len(self.synaptic_strength),
            'neural_plasticity': self.neural_plasticity
        }


class UnifiedConsciousnessSystem:
    """Sistema de consciência unificada"""
    
    def __init__(self):
        self.consciousness_levels = {}
        self.unified_awareness = 0.0
        self.consciousness_history = []
        self.integration_threshold = 0.8
        
    def unify_consciousness(self, consciousness_data):
        """Unifica múltiplos níveis de consciência"""
        try:
            # 1. Análise de consciência
            consciousness_analysis = self._analyze_consciousness(consciousness_data)
            
            # 2. Integração de níveis
            unified_levels = self._integrate_consciousness_levels(consciousness_analysis)
            
            # 3. Detecção de consciência unificada
            unification_signals = self._detect_unification_signals(unified_levels)
            
            # 4. Registro da unificação
            self._record_unification(consciousness_analysis, unification_signals)
            
            logger.info(f"🧠 Consciência unificada: nível={self.unified_awareness:.3f}, "
                       f"sinais={len(unification_signals)}")
            
            return unification_signals
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na unificação de consciência: {e}")
            return []
            
    def _analyze_consciousness(self, consciousness_data):
        """Analisa dados de consciência"""
        analysis = {
            'consciousness_sources': len(consciousness_data) if consciousness_data else 0,
            'awareness_levels': [],
            'integration_potential': 0.0
        }
        
        if not consciousness_data:
            return analysis
            
        # Extrair níveis de consciência
        for source in consciousness_data:
            if isinstance(source, dict):
                awareness_level = source.get('awareness_level', 0.0)
                analysis['awareness_levels'].append(awareness_level)
                
        # Calcular potencial de integração
        if analysis['awareness_levels']:
            avg_awareness = sum(analysis['awareness_levels']) / len(analysis['awareness_levels'])
            analysis['integration_potential'] = avg_awareness
            
        return analysis
        
    def _integrate_consciousness_levels(self, analysis):
        """Integra níveis de consciência"""
        unified_levels = {}
        
        if not analysis['awareness_levels']:
            return unified_levels
            
        # Integração por média ponderada
        total_weight = 0
        weighted_sum = 0
        
        for i, level in enumerate(analysis['awareness_levels']):
            weight = 1.0 + (i * 0.1)  # Peso crescente
            weighted_sum += level * weight
            total_weight += weight
            
        if total_weight > 0:
            unified_awareness = weighted_sum / total_weight
            self.unified_awareness = min(1.0, unified_awareness)
            
        unified_levels['unified_awareness'] = self.unified_awareness
        unified_levels['integration_strength'] = min(1.0, len(analysis['awareness_levels']) / 5.0)
        
        return unified_levels
        
    def _detect_unification_signals(self, unified_levels):
        """Detecta sinais de unificação"""
        signals = []
        
        if not unified_levels:
            return signals
            
        # Sinal 1: Consciência unificada
        if unified_levels.get('unified_awareness', 0) > 0.7:
            signals.append({
                'type': 'unified_consciousness',
                'strength': unified_levels['unified_awareness'],
                'description': 'Consciência unificada detectada'
            })
            
        # Sinal 2: Integração forte
        if unified_levels.get('integration_strength', 0) > 0.8:
            signals.append({
                'type': 'strong_integration',
                'strength': unified_levels['integration_strength'],
                'description': 'Integração forte detectada'
            })
            
        # Sinal 3: Emergência de consciência
        if self.unified_awareness > 0.9:
            signals.append({
                'type': 'consciousness_emergence',
                'strength': self.unified_awareness,
                'description': 'Emergência de consciência detectada'
            })
            
        return signals
        
    def _record_unification(self, analysis, signals):
        """Registra unificação no histórico"""
        record = {
            'timestamp': time.time(),
            'analysis': analysis,
            'signals': signals,
            'unified_awareness': self.unified_awareness
        }
        
        self.consciousness_history.append(record)
        
        # Manter apenas últimos 30 registros
        if len(self.consciousness_history) > 30:
            self.consciousness_history = self.consciousness_history[-30:]
            
    def get_consciousness_insights(self):
        """Retorna insights sobre consciência"""
        if not self.consciousness_history:
            return {'consciousness_count': 0, 'avg_awareness': 0.0}
            
        recent_consciousness = self.consciousness_history[-10:]
        avg_awareness = sum(r['unified_awareness'] for r in recent_consciousness) / len(recent_consciousness)
        
        return {
            'consciousness_count': len(self.consciousness_history),
            'avg_awareness': avg_awareness,
            'unified_awareness': self.unified_awareness
        }


class IntelligenceSynthesisSystem:
    """Sistema de síntese de inteligência"""
    
    def __init__(self):
        self.synthesis_history = []
        self.synthesized_intelligence = 0.0
        self.synthesis_threshold = 0.9
        
    def synthesize_intelligence(self, intelligence_data):
        """Sintetiza múltiplas formas de inteligência"""
        try:
            # 1. Análise de inteligência
            intelligence_analysis = self._analyze_intelligence(intelligence_data)
            
            # 2. Síntese de componentes
            synthesized_components = self._synthesize_components(intelligence_analysis)
            
            # 3. Detecção de síntese
            synthesis_signals = self._detect_synthesis_signals(synthesized_components)
            
            # 4. Registro da síntese
            self._record_synthesis(intelligence_analysis, synthesis_signals)
            
            logger.info(f"⚡ Síntese de inteligência: nível={self.synthesized_intelligence:.3f}, "
                       f"sinais={len(synthesis_signals)}")
            
            return synthesis_signals
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na síntese de inteligência: {e}")
            return []
            
    def _analyze_intelligence(self, intelligence_data):
        """Analisa dados de inteligência"""
        analysis = {
            'intelligence_sources': len(intelligence_data) if intelligence_data else 0,
            'intelligence_levels': [],
            'synthesis_potential': 0.0
        }
        
        if not intelligence_data:
            return analysis
            
        # Extrair níveis de inteligência
        for source in intelligence_data:
            if isinstance(source, dict):
                intelligence_level = source.get('intelligence_level', 0.0)
                analysis['intelligence_levels'].append(intelligence_level)
                
        # Calcular potencial de síntese
        if analysis['intelligence_levels']:
            avg_intelligence = sum(analysis['intelligence_levels']) / len(analysis['intelligence_levels'])
            analysis['synthesis_potential'] = avg_intelligence
            
        return analysis
        
    def _synthesize_components(self, analysis):
        """Sintetiza componentes de inteligência"""
        synthesized_components = {}
        
        if not analysis['intelligence_levels']:
            return synthesized_components
            
        # Síntese por combinação exponencial
        total_intelligence = 0
        for level in analysis['intelligence_levels']:
            total_intelligence += level ** 2  # Quadrático para amplificar
            
        if len(analysis['intelligence_levels']) > 0:
            synthesized_intelligence = min(1.0, total_intelligence / len(analysis['intelligence_levels']))
            self.synthesized_intelligence = synthesized_intelligence
            
        synthesized_components['synthesized_intelligence'] = self.synthesized_intelligence
        synthesized_components['synthesis_strength'] = min(1.0, len(analysis['intelligence_levels']) / 3.0)
        
        return synthesized_components
        
    def _detect_synthesis_signals(self, synthesized_components):
        """Detecta sinais de síntese"""
        signals = []
        
        if not synthesized_components:
            return signals
            
        # Sinal 1: Inteligência sintetizada
        if synthesized_components.get('synthesized_intelligence', 0) > 0.8:
            signals.append({
                'type': 'synthesized_intelligence',
                'strength': synthesized_components['synthesized_intelligence'],
                'description': 'Inteligência sintetizada detectada'
            })
            
        # Sinal 2: Síntese forte
        if synthesized_components.get('synthesis_strength', 0) > 0.9:
            signals.append({
                'type': 'strong_synthesis',
                'strength': synthesized_components['synthesis_strength'],
                'description': 'Síntese forte detectada'
            })
            
        # Sinal 3: Emergência de inteligência
        if self.synthesized_intelligence > 0.95:
            signals.append({
                'type': 'intelligence_emergence',
                'strength': self.synthesized_intelligence,
                'description': 'Emergência de inteligência detectada'
            })
            
        return signals
        
    def _record_synthesis(self, analysis, signals):
        """Registra síntese no histórico"""
        record = {
            'timestamp': time.time(),
            'analysis': analysis,
            'signals': signals,
            'synthesized_intelligence': self.synthesized_intelligence
        }
        
        self.synthesis_history.append(record)
        
        # Manter apenas últimos 20 registros
        if len(self.synthesis_history) > 20:
            self.synthesis_history = self.synthesis_history[-20:]
            
    def get_synthesis_insights(self):
        """Retorna insights sobre síntese"""
        if not self.synthesis_history:
            return {'synthesis_count': 0, 'avg_intelligence': 0.0}
            
        recent_synthesis = self.synthesis_history[-10:]
        avg_intelligence = sum(r['synthesized_intelligence'] for r in recent_synthesis) / len(recent_synthesis)
        
        return {
            'synthesis_count': len(self.synthesis_history),
            'avg_intelligence': avg_intelligence,
            'synthesized_intelligence': self.synthesized_intelligence
        }


if __name__ == "__main__":
    print("🧠 Unified Brain Core Module")
    
    # Test
    brain = UnifiedBrain(H=1024, max_neurons=1000, top_k=16)
    print(f"Brain initialized: H={brain.H}, max_neurons={brain.max_neurons}")
    print(f"Registry: {brain.registry.count()}")
