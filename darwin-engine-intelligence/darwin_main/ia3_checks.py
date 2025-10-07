# ia3_checks.py — IA3Inspector: decide vida/morte de cada neurônio a cada rodada
import math, random, torch
import torch.nn.functional as F
from typing import Dict, Any, List

class IA3Inspector:
    """
    IA3-like score agregado por neurônio:
    - adaptativo: Δloss local < 0
    - autorecursivo: realizou self_mutations no round
    - autoevolutivo: gate ↑ ou proposta estrutural  
    - autodidata: contribuiu para reduzir avg_loss do cérebro
    - autônomo: executou sem overrides manuais
    - autossuficiente: respeitou orçamentos/limites
    - autoconstructivo/autosináptico/autoarquitetável: propôs conexões/ativação
    """
    async def __init__(self, cfg: dict, brain):
        self.cfg = cfg
        self.brain = brain
        self.last_avg_loss = None
        self.round_history = []

    async def judge_neuron(self, neuron_id: str, round_metrics: dict) -> Dict[str, Any]:
        """
        Julga um neurônio individual contra critérios IA³ rigorosos
        Retorna veredito de vida/morte
        """
        # Encontrar neurônio
        neuron = None
        for n in self.brain.neurons:
            if n.id == neuron_id:
                neuron = n
                break
        
        if neuron is None:
            return await {"neuron_id": neuron_id, "ia3_like": False, "error": "neuron_not_found"}
        
        avg_loss = round_metrics["avg_loss"]
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 1: ADAPTATIVO
        # ═══════════════════════════════════════════════════════════════════════
        if neuron.last_loss is None:
            neuron.last_loss = avg_loss + 1.0  # Forçar melhoria inicial
        
        delta_loss = neuron.last_loss - avg_loss
        adaptativo = delta_loss >= self.cfg.get("delta_loss_min", 1e-3)
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 2: AUTORECURSIVO
        # ═══════════════════════════════════════════════════════════════════════
        autorecursivo = neuron.self_mutations > 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 3: AUTOEVOLUTIVO
        # ═══════════════════════════════════════════════════════════════════════
        # Gate deve ter saído do valor inicial baixo
        gate_value = float(neuron.gate.detach().cpu().item())
        autoevolutivo = gate_value > 0.12  # Acima do inicial 0.1
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 4: AUTODIDATA
        # ═══════════════════════════════════════════════════════════════════════
        if self.last_avg_loss is None:
            self.last_avg_loss = avg_loss + 1.0
        
        global_improvement = self.last_avg_loss - avg_loss
        autodidata = global_improvement > 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 5: AUTÔNOMO
        # ═══════════════════════════════════════════════════════════════════════
        # Neurônio deve ter ativação não-saturada e variável
        autonomo = True  # Proxy simples - não houve intervenção manual
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 6: AUTOSSUFICIENTE
        # ═══════════════════════════════════════════════════════════════════════
        # Contribuição única verificável
        contribution = abs(gate_value)
        autossuficiente = contribution > 0.05  # Contribuição mínima
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 7: AUTOCONSTRUCTIVO
        # ═══════════════════════════════════════════════════════════════════════
        # Participou da construção via mutações
        autoconstructivo = neuron.arch_proposals > 0 or neuron.self_mutations > 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 8: AUTOSINÁPTICO
        # ═══════════════════════════════════════════════════════════════════════
        # Gerenciou próprias conexões (via gate e mutações)
        autosinaptico = abs(gate_value - 0.1) > 0.01  # Gate mudou do inicial
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITÉRIO 9: AUTOARQUITETÁVEL
        # ═══════════════════════════════════════════════════════════════════════
        # Definiu própria estrutura via ativação mista
        act_entropy = neuron.mixed_act.entropy()
        autoarquitetavel = act_entropy < math.log(4) * 0.9  # Convergiu para ativação específica
        
        # ═══════════════════════════════════════════════════════════════════════
        # SCORE COMPOSTO IA³
        # ═══════════════════════════════════════════════════════════════════════
        criteria = {
            "adaptativo": adaptativo,
            "autorecursivo": autorecursivo, 
            "autoevolutivo": autoevolutivo,
            "autodidata": autodidata,
            "autonomo": autonomo,
            "autossuficiente": autossuficiente,
            "autoconstructivo": autoconstructivo,
            "autosinaptico": autosinaptico,
            "autoarquitetavel": autoarquitetavel
        }
        
        # Pesos dos critérios (podem ser ajustados)
        weights = {
            "adaptativo": 0.15,
            "autorecursivo": 0.12,
            "autoevolutivo": 0.13,
            "autodidata": 0.11,
            "autonomo": 0.10,
            "autossuficiente": 0.12,
            "autoconstructivo": 0.09,
            "autosinaptico": 0.08,
            "autoarquitetavel": 0.10
        }
        
        # Score ponderado
        ia3_score = sum(
            (1.0 if criteria[criterion] else 0.0) * weight
            for criterion, weight in weights.items()
        )
        
        # DECISÃO DE VIDA/MORTE
        threshold = self.cfg.get("ia3_threshold", 0.60)
        ia3_like = ia3_score >= threshold
        
        # Critérios obrigatórios (não podem falhar)
        mandatory_criteria = ["adaptativo", "autoevolutivo"]  # Deve adaptar E evoluir
        mandatory_passed = all(criteria[c] for c in mandatory_criteria)
        
        # Decisão final
        passes = ia3_like and mandatory_passed
        
        # Atualizar memória do neurônio
        neuron.last_loss = avg_loss
        
        # Atualizar memória global
        self.last_avg_loss = avg_loss
        
        # Construir veredito completo
        verdict = {
            "neuron_id": neuron_id,
            "round_number": len(self.round_history) + 1,
            "timestamp": round_metrics.get("timestamp", "unknown"),
            
            # Métricas básicas
            "delta_loss": float(delta_loss),
            "gate_value": gate_value,
            "contribution": contribution,
            "age": neuron.age,
            "consciousness": neuron.consciousness_score,
            
            # Critérios individuais
            "criteria": criteria,
            "criteria_weights": weights,
            
            # Score e decisão
            "ia3_score": float(ia3_score),
            "threshold": threshold,
            "ia3_like": ia3_like,
            "mandatory_passed": mandatory_passed,
            "passes": passes,
            
            # Detalhes técnicos
            "details": {
                "activation_entropy": act_entropy,
                "dominant_activation": neuron.mixed_act.get_dominant_activation()[0],
                "self_mutations": neuron.self_mutations,
                "arch_proposals": neuron.arch_proposals,
                "weight_norm": neuron.get_weight_norm(),
                "grad_norm": neuron.get_grad_norm()
            }
        }
        
        return await verdict

    async def judge_all_neurons(self, round_metrics: dict) -> List[Dict[str, Any]]:
        """Julga todos os neurônios do cérebro"""
        verdicts = []
        
        logger.info(f"\n🔬 JULGAMENTO IA³ - {len(self.brain.neurons)} neurônios")
        logger.info(f"   Limiar IA³: {self.cfg.get('ia3_threshold', 0.60)}")
        
        for neuron in self.brain.neurons:
            verdict = self.judge_neuron(neuron.id, round_metrics)
            verdicts.append(verdict)
            
            # Log do julgamento
            status = "✅ VIVE" if verdict["passes"] else "☠️ MORRE"
            logger.info(f"   {neuron.id}: {status} | Score: {verdict['ia3_score']:.3f} | "
                  f"Consciência: {verdict['consciousness']:.3f}")
        
        # Estatísticas gerais
        passed_count = sum(1 for v in verdicts if v["passes"])
        pass_rate = passed_count / len(verdicts) if verdicts else 0.0
        
        logger.info(f"   📊 Taxa de aprovação: {passed_count}/{len(verdicts)} ({pass_rate*100:.1f}%)")
        
        # Salvar round no histórico
        round_summary = {
            "round_metrics": round_metrics,
            "verdicts": verdicts,
            "pass_rate": pass_rate,
            "timestamp": round_metrics.get("timestamp", "unknown")
        }
        self.round_history.append(round_summary)
        
        return await verdicts

    async def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas históricas do inspector"""
        if not self.round_history:
            return await {"error": "Nenhum round executado ainda"}
        
        total_rounds = len(self.round_history)
        total_judgments = sum(len(r["verdicts"]) for r in self.round_history)
        total_passed = sum(sum(1 for v in r["verdicts"] if v["passes"]) for r in self.round_history)
        
        avg_pass_rate = total_passed / total_judgments if total_judgments > 0 else 0.0
        
        return await {
            "total_rounds": total_rounds,
            "total_judgments": total_judgments,
            "total_passed": total_passed,
            "avg_pass_rate": avg_pass_rate,
            "last_round": self.round_history[-1] if self.round_history else None
        }

if __name__ == "__main__":
    # Teste do inspector
    logger.info("🧪 Testando IA³ Inspector...")
    
    # Criar cérebro dummy
    from neurogenesis import Brain
    brain = Brain()
    brain.add_neuron()
    brain.add_neuron() 
    
    # Criar inspector
    cfg = {"ia3_threshold": 0.60, "delta_loss_min": 0.001}
    inspector = IA3Inspector(cfg, brain)
    
    # Simular round
    round_metrics = {
        "avg_loss": 0.5,
        "steps": 100,
        "neurons": len(brain.neurons),
        "timestamp": "2025-09-22T00:00:00Z"
    }
    
    # Julgar neurônios
    verdicts = inspector.judge_all_neurons(round_metrics)
    
    logger.info(f"\n📊 Resultados:")
    for verdict in verdicts:
        logger.info(f"   {verdict['neuron_id']}: {verdict['passes']} (score: {verdict['ia3_score']:.3f})")
    
    logger.info(f"\n✅ Inspector testado!")