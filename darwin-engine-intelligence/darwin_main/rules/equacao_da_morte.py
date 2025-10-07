# /root/darwin/rules/equacao_da_morte.py
# Equação da Morte Darwin v4: E(t+1) = A(t) OR C(t)
# Vida só se o neurônio provar ser IA³-like a cada rodada

from dataclasses import dataclass
from typing import Dict, Any
import math

@dataclass
class DeathEquationConfig:
    """Configuração da Equação da Morte Darwin v4"""
    BETA_MIN: float = 0.01      # melhoria mínima de loss exigida
    NOVELTY_TAU: float = 0.05   # limiar de novidade estrutural
    CONSCIOUSNESS_MIN: float = 0.3  # consciência mínima exigida
    ADAPTATION_MIN: float = 0.001   # adaptação mínima em OOD
    EVOLUTION_MIN: float = 0.01     # evolução estrutural mínima
    
    # Pesos dos critérios IA³
    WEIGHTS: Dict[str, float] = None
    
    async def __post_init__(self):
        if self.WEIGHTS is None:
            self.WEIGHTS = {
                "adaptativo": 0.15,
                "autorecursivo": 0.12,
                "autoevolutivo": 0.13,
                "autodidata": 0.11,
                "autonomo": 0.12,
                "autossuficiente": 0.10,
                "autoconstructivo": 0.09,
                "autosinaptico": 0.08,
                "autoarquitetavel": 0.10
            }

async def calculate_ia3_criteria(metrics: Dict[str, Any], cfg: DeathEquationConfig) -> Dict[str, bool]:
    """
    Calcula cada critério IA³ baseado nas métricas do neurônio
    """
    criteria = {}
    
    # 1. ADAPTATIVO - Melhora em loss de validação
    delta_loss = metrics.get("delta_val_loss", 0.0)
    criteria["adaptativo"] = delta_loss <= -cfg.BETA_MIN
    
    # 2. AUTORECURSIVO - Realizou auto-mutações
    self_mutations = metrics.get("self_mutations", 0)
    criteria["autorecursivo"] = self_mutations > 0
    
    # 3. AUTOEVOLUTIVO - Produziu mudança estrutural
    novelty = metrics.get("novelty", 0.0)
    evolution_success = metrics.get("self_evolve_success", False)
    criteria["autoevolutivo"] = novelty >= cfg.EVOLUTION_MIN and evolution_success
    
    # 4. AUTODIDATA - Aprendeu via auto-supervisão
    learning_improvement = abs(delta_loss) > cfg.BETA_MIN
    criteria["autodidata"] = learning_improvement
    
    # 5. AUTÔNOMO - Operou sem intervenção externa
    autonomous = metrics.get("autonomous_operation", True)
    criteria["autonomo"] = autonomous
    
    # 6. AUTOSSUFICIENTE - Contribuição positiva verificável
    contribution = metrics.get("contribution_score", 0.0)
    criteria["autossuficiente"] = contribution > 0.1
    
    # 7. AUTOCONSTRUCTIVO - Participou da própria construção
    construction_activity = metrics.get("construction_mutations", 0)
    criteria["autoconstructivo"] = construction_activity > 0 or evolution_success
    
    # 8. AUTOSINÁPTICO - Gerenciou conexões sinápticas
    synaptic_activity = metrics.get("synaptic_changes", 0)
    criteria["autosinaptico"] = synaptic_activity > 0 or self_mutations > 0
    
    # 9. AUTOARQUITETÁVEL - Definiu própria arquitetura
    architectural_changes = metrics.get("architectural_changes", 0)
    mutation_success = metrics.get("best_mutation") is not None
    criteria["autoarquitetavel"] = architectural_changes > 0 or mutation_success
    
    return await criteria

async def calculate_consciousness_score(metrics: Dict[str, Any], criteria: Dict[str, bool]) -> float:
    """
    Calcula score de consciência neural baseado em múltiplos fatores
    """
    # Base: proporção de critérios IA³ atendidos
    base_score = sum(criteria.values()) / len(criteria)
    
    # Fatores adicionais
    novelty = metrics.get("novelty", 0.0)
    adaptation = abs(metrics.get("delta_val_loss", 0.0))
    complexity = metrics.get("architectural_complexity", 0.0)
    
    # Fórmula composta
    consciousness = (
        0.4 * base_score +                    # Critérios IA³
        0.2 * min(1.0, novelty * 10) +       # Novidade
        0.2 * min(1.0, adaptation * 100) +   # Adaptação
        0.2 * min(1.0, complexity)           # Complexidade
    )
    
    return await max(0.0, min(1.0, consciousness))

async def equacao_da_morte(metrics: Dict[str, Any], cfg: DeathEquationConfig = None) -> Dict[str, Any]:
    """
    Equação da Morte Darwin v4: E(t+1) = A(t) OR C(t)
    
    A(t) = Auto-evolução comprovada (todos os critérios IA³ básicos)
    C(t) = Descoberta generalizável (novidade alta + melhoria)
    
    Retorna: dicionário com decisão de vida/morte e detalhes
    """
    if cfg is None:
        cfg = DeathEquationConfig()
    
    # Calcular critérios IA³
    criteria = calculate_ia3_criteria(metrics, cfg)
    
    # Calcular scores
    delta_loss = metrics.get("delta_val_loss", 0.0)
    novelty = metrics.get("novelty", 0.0)
    self_evolve_success = metrics.get("self_evolve_success", False)
    
    # A(t): Auto-evolução comprovada
    # Deve melhorar E ter sucesso na evolução E passar nos critérios essenciais
    essential_criteria = ["adaptativo", "autoevolutivo", "autodidata"]
    A_criteria_met = all(criteria.get(c, False) for c in essential_criteria)
    A = (delta_loss <= -cfg.BETA_MIN) and self_evolve_success and A_criteria_met
    
    # C(t): Descoberta generalizável
    # Novidade alta E alguma melhoria (mesmo que pequena)
    C = (novelty >= cfg.NOVELTY_TAU) and (delta_loss <= 0.0) and self_evolve_success
    
    # E(t+1): Decisão final
    E = 1 if (A or C) else 0
    
    # Score composto IA³
    ia3_score = sum(
        (1.0 if criteria[criterion] else 0.0) * weight
        for criterion, weight in cfg.WEIGHTS.items()
        if criterion in criteria
    )
    
    # Consciência neural
    consciousness = calculate_consciousness_score(metrics, criteria)
    
    # Razões da decisão
    reasons = []
    if A:
        reasons.append("A(t)=1: Auto-evolução comprovada")
    if C:
        reasons.append("C(t)=1: Descoberta generalizável")
    if not A and not C:
        failed_criteria = [c for c, passed in criteria.items() if not passed]
        reasons.append(f"A=0 e C=0: Falhou em {failed_criteria}")
    
    return await {
        "E": E,
        "decision": "VIVE" if E == 1 else "MORRE",
        "A": A,
        "C": C,
        "ia3_score": ia3_score,
        "consciousness_score": consciousness,
        "criteria": criteria,
        "metrics_used": {
            "delta_val_loss": delta_loss,
            "novelty": novelty,
            "self_evolve_success": self_evolve_success
        },
        "reasons": reasons,
        "config": {
            "beta_min": cfg.BETA_MIN,
            "novelty_tau": cfg.NOVELTY_TAU,
            "consciousness_min": cfg.CONSCIOUSNESS_MIN
        }
    }

async def is_extinction_necessary(death_streak: int, max_deaths_without_birth: int = 50) -> bool:
    """
    Determina se é necessário reiniciar completamente (extinção total)
    """
    return await death_streak >= max_deaths_without_birth

async def calculate_heritage(survivors_history: list, deaths_history: list) -> Dict[str, Any]:
    """
    Calcula herança para novos neurônios baseada em histórico
    """
    if not survivors_history and not deaths_history:
        return await {"strategy": "random_initialization"}
    
    # Herança positiva (dos sobreviventes)
    positive_traits = {}
    if survivors_history:
        # Média das melhores configurações
        best_configs = [s.get("config", {}) for s in survivors_history[-10:]]  # Últimos 10
        if best_configs:
            # Extrair patterns comuns
            positive_traits = {
                "preferred_lr": sum(c.get("lr", 0.01) for c in best_configs) / len(best_configs),
                "successful_activations": [c.get("act", "relu") for c in best_configs],
                "avg_novelty": sum(c.get("novelty", 0) for c in best_configs) / len(best_configs)
            }
    
    # Herança negativa (dos mortos - o que evitar)
    negative_traits = {}
    if deaths_history:
        recent_deaths = deaths_history[-20:]  # Últimas 20 mortes
        failed_patterns = {}
        for death in recent_deaths:
            failure_reason = death.get("reason", "unknown")
            failed_patterns[failure_reason] = failed_patterns.get(failure_reason, 0) + 1
        
        negative_traits = {
            "common_failures": failed_patterns,
            "avoided_configs": [d.get("config", {}) for d in recent_deaths]
        }
    
    return await {
        "strategy": "learned_heritage",
        "positive_traits": positive_traits,
        "negative_traits": negative_traits,
        "generation": len(survivors_history) + len(deaths_history)
    }

if __name__ == "__main__":
    import sys
    
    # Teste da Equação da Morte
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logger.info("🧪 Testando Equação da Morte Darwin v4...")
        
        # Caso 1: Neurônio que sobrevive (A=1)
        metrics_survive = {
            "delta_val_loss": -0.05,  # Melhorou
            "novelty": 0.03,
            "self_evolve_success": True,
            "self_mutations": 3,
            "contribution_score": 0.2,
            "architectural_changes": 1
        }
        
        result1 = equacao_da_morte(metrics_survive)
        logger.info(f"\n📊 Teste 1 - Neurônio que deveria VIVER:")
        logger.info(f"   Decisão: {result1['decision']}")
        logger.info(f"   A(t): {result1['A']}, C(t): {result1['C']}")
        logger.info(f"   Score IA³: {result1['ia3_score']:.3f}")
        logger.info(f"   Consciência: {result1['consciousness_score']:.3f}")
        
        # Caso 2: Neurônio que morre (A=0, C=0)
        metrics_die = {
            "delta_val_loss": 0.02,   # Piorou
            "novelty": 0.01,          # Baixa novidade
            "self_evolve_success": False,
            "self_mutations": 0,
            "contribution_score": 0.05,
            "architectural_changes": 0
        }
        
        result2 = equacao_da_morte(metrics_die)
        logger.info(f"\n📊 Teste 2 - Neurônio que deveria MORRER:")
        logger.info(f"   Decisão: {result2['decision']}")
        logger.info(f"   A(t): {result2['A']}, C(t): {result2['C']}")
        logger.info(f"   Score IA³: {result2['ia3_score']:.3f}")
        logger.info(f"   Consciência: {result2['consciousness_score']:.3f}")
        logger.info(f"   Razões: {result2['reasons']}")
        
        logger.info(f"\n✅ Equação da Morte funcionando!")
    else:
        logger.info("Uso: python equacao_da_morte.py --test")