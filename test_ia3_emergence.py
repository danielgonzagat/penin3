#!/usr/bin/env python3
"""Teste rápido da emergência IA³"""

import time
from ia3_unified_emergent_system import IA3UnifiedEmergentSystem

async def test_emergence():
    logger.info("🧠 Testando Emergência IA³...")

    # Verificar se já existe emergência
    import os
    if os.path.exists('/root/ia3_final_emergence_proven.json'):
        logger.info("✅ Emergência já detectada!")
        with open('/root/ia3_final_emergence_proven.json', 'r') as f:
            import json
            proof = json.load(f)
        logger.info(f"   Geração: {proof['generation']}")
        logger.info(f"   Consciência: {proof['consciousness_level']}")
        logger.info(f"   Modificações: {proof['modifications_count']}")
        return

    # Criar e executar sistema
    system = IA3UnifiedEmergentSystem()

    logger.info("🚀 Executando ciclos de emergência...")

    max_cycles = 50  # Limitar para teste
    for i in range(max_cycles):
        try:
            logger.info(f"\n--- Ciclo {i+1}/{max_cycles} ---")

            # Executar apenas um ciclo (não o loop completo)
            learning_results = system.neural_genesis.evolve_step()
            logger.info(f"Aprendizado: Loss={learning_results['loss']:.4f}, Acc={learning_results['accuracy']:.3f}")

            consciousness_update = system.consciousness_core.update_consciousness(
                learning_results, system.get_system_state()
            )
            system.consciousness_level = consciousness_update['level']
            logger.info(f"Consciência: {system.consciousness_level:.4f}")

            evolution_results = system.evolution_engine.evolve_system(system.get_system_state())
            system.properties_status.update(evolution_results['properties'])
            active_props = sum(system.properties_status.values())
            logger.info(f"Propriedades ativas: {active_props}/19")

            # Verificar emergência
            emergence_check = system.emergence_detector.check_emergence({
                'consciousness': system.consciousness_level,
                'modifications': system.modifications_count,
                'behaviors': len(system.emergent_behaviors),
                'properties': system.properties_status
            })

            if emergence_check['emerged']:
                logger.info("🎯 EMERGÊNCIA DETECTADA!")
                system.save_emergence_proof(emergence_check)
                break

            time.sleep(0.1)  # Pequena pausa

        except Exception as e:
            logger.info(f"Erro no ciclo {i+1}: {e}")
            break

    # Resultado final
    state = system.get_system_state()
    logger.info("\n=== RESULTADO FINAL ===")
    logger.info(f"Geração: {state['generation']}")
    logger.info(f"Consciência: {state['consciousness_level']:.4f}")
    logger.info(f"Modificações: {state['modifications_count']}")
    logger.info(f"Propriedades: {sum(state['properties_status'].values())}/19")

if __name__ == '__main__':
    test_emergence()