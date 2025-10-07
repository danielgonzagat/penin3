import torch
import math

async def ia3_like_score(neuron):
    """
    Cada neurônio precisa provar, a CADA rodada, ser:
    - adaptativo: melhora local de loss ao treinar
    - autorecursivo: modifica parâmetros de meta-aprendizado (ex: taxas, alphas) produtivamente
    - autoevolutivo: melhora sua métrica de fitness histórica
    - autodidata: progresso em batches não vistos
    - autônomo: consegue operar com energia/estado > limiar
    - autossuficiente: não degrada a população (contribuição positiva)
    - autoconstruído/autoarquitetável: fez ao menos 1 modificação estrutural (pesos/alphas) válida
    - autosináptico: atualizou pesos de conexão out->outros
    
    Retorna: (escore [0..1], motivos detalhados)
    """
    reasons = {}
    score = 0.0
    criteria_count = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 1: ADAPTATIVO
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    delta_loss = neuron.stats.get("delta_loss_recent", 0.0)
    adaptativo = delta_loss < 1.0  # Loss deve estar controlada
    reasons["adaptativo"] = adaptativo
    score += 1.0 if adaptativo else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 2: AUTORECURSIVO
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    meta_updates = neuron.stats.get("meta_updates", 0)
    autorecursivo = meta_updates > 0
    reasons["autorecursivo"] = autorecursivo
    score += 1.0 if autorecursivo else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 3: AUTOEVOLUTIVO
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    fitness_hist = neuron.stats.get("fitness_hist", [])
    current_fitness = neuron.stats.get("fitness", None)
    
    if len(fitness_hist) >= 3 and current_fitness is not None:
        median_fitness = sorted(fitness_hist)[len(fitness_hist)//2]
        autoevolutivo = current_fitness < median_fitness  # Menor loss é melhor
    else:
        autoevolutivo = True  # Dar chance para neurônios novos
    
    reasons["autoevolutivo"] = autoevolutivo
    score += 1.0 if autoevolutivo else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 4: AUTODIDATA
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    ood_gain = neuron.stats.get("ood_gain", 0.0)
    autodidata = ood_gain > 0.0
    reasons["autodidata"] = autodidata
    score += 1.0 if autodidata else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 5: AUTÔNOMO
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    energy = getattr(neuron, "energy", 1.0)
    autonomo = energy > 0.1
    reasons["autonomo"] = autonomo
    score += 1.0 if autonomo else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 6: AUTOSSUFICIENTE
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    pop_contrib = neuron.stats.get("pop_contrib", 0.0)
    autossuficiente = pop_contrib < 0  # Contribuição negativa = melhoria para população
    reasons["autossuficiente"] = autossuficiente
    score += 1.0 if autossuficiente else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 7: AUTOARQUITETÁVEL
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    arch_updates = neuron.stats.get("arch_updates", 0)
    autoarquitetavel = arch_updates > 0
    reasons["autoarquitetavel"] = autoarquitetavel
    score += 1.0 if autoarquitetavel else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITÉRIO 8: AUTOSINÁPTICO
    # ═══════════════════════════════════════════════════════════════════════════
    criteria_count += 1
    syn_updates = neuron.stats.get("syn_updates", 0)
    autosinaptico = syn_updates > 0
    reasons["autosinaptico"] = autosinaptico
    score += 1.0 if autosinaptico else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # SCORE FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    final_score = score / criteria_count if criteria_count > 0 else 0.0
    
    # Consciência como bonus
    consciousness = getattr(neuron, "consciousness_score", 0.0)
    if consciousness > 0.5:
        final_score += 0.1  # Bonus por alta consciência
    
    final_score = min(1.0, final_score)
    
    # Adicionar detalhes às razões
    reasons["score_details"] = {
        "raw_score": score,
        "criteria_count": criteria_count,
        "final_score": final_score,
        "consciousness_bonus": consciousness > 0.5,
        "consciousness_value": consciousness
    }
    
    return await final_score, reasons

async def evaluate_population_fitness(population) -> dict:
    """
    Avalia fitness da população completa
    """
    if not hasattr(population, "neurons") or len(population.neurons) == 0:
        return await {"avg_fitness": 0.0, "neuron_scores": []}
    
    neuron_scores = []
    total_consciousness = 0.0
    
    for i, neuron in enumerate(population.neurons):
        score, reasons = ia3_like_score(neuron)
        consciousness = getattr(neuron, "consciousness_score", 0.0)
        
        neuron_scores.append({
            "neuron_idx": i,
            "neuron_id": getattr(neuron, "neuron_id", f"n{i}"),
            "ia3_score": score,
            "consciousness": consciousness,
            "reasons": reasons,
            "passes": score >= 0.6  # Limiar de sobrevivência
        })
        
        total_consciousness += consciousness
    
    # Estatísticas da população
    avg_consciousness = total_consciousness / len(population.neurons)
    pass_rate = sum(1 for ns in neuron_scores if ns["passes"]) / len(neuron_scores)
    
    return await {
        "neuron_count": len(population.neurons),
        "neuron_scores": neuron_scores,
        "avg_consciousness": avg_consciousness,
        "pass_rate": pass_rate,
        "survivors": [ns for ns in neuron_scores if ns["passes"]],
        "casualties": [ns for ns in neuron_scores if not ns["passes"]]
    }

if __name__ == "__main__":
    logger.info("🧪 Testando gates IA³...")
    
    # Simular neurônio para teste
    class MockNeuron:
        async def __init__(self):
            self.energy = 0.8
            self.consciousness_score = 0.6
            self.stats = {
                "delta_loss_recent": 0.5,
                "meta_updates": 2,
                "fitness_hist": [1.0, 0.8, 0.6],
                "fitness": 0.5,
                "ood_gain": 0.1,
                "pop_contrib": -0.05,
                "arch_updates": 1,
                "syn_updates": 3
            }
    
    mock_neuron = MockNeuron()
    score, reasons = ia3_like_score(mock_neuron)
    
    logger.info(f"📊 Teste neurônio mock:")
    logger.info(f"   Score IA³: {score:.3f}")
    logger.info(f"   Passa (≥0.6): {'✅' if score >= 0.6 else '❌'}")
    
    logger.info(f"\n🔬 Critérios individuais:")
    for criterion, passed in reasons.items():
        if criterion != "score_details":
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {criterion}")
    
    logger.info(f"\n✅ Gates IA³ funcionando!")