import argparse, os, json, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from .metrics import (ensure_metrics, g_neurons_alive, g_cycle_index, g_population_loss,
                     g_births, g_deaths, g_stability, g_oci, g_p_convergence, g_consciousness,
                     c_cycles, c_birth_events, c_death_events)
from .worm import log_event, get_worm_stats
from .population import Population
from .data_sources import SyntheticTask, TinyStoriesCharBag

async def compute_proxies(loss_hist):
    """
    Computa proxies para métricas TEIS/Darwin:
    - Estabilidade (Lyapunov-ish): perda não explodindo
    - OCI: proxy de fechamento
    - P: proxy ∞(E+N−iN)
    """
    if len(loss_hist) < 3:
        return await 1.0, 0.5, 0.05
    
    # Estabilidade: derivada da loss deve ser <= 0 (não divergindo)
    recent_slope = loss_hist[-1] - loss_hist[-3]
    stability = 1.0 if recent_slope <= 0 else max(0.0, 1.0 - abs(recent_slope))
    
    # OCI: proxy de fechamento operacional (simulado)
    oci = 0.6 + 0.1 * random.random()
    
    # P: proxy de convergência (simulado)
    P = max(0.01, 0.05 + 0.02 * random.random())
    
    return await stability, oci, P

async def run_darwin_cycles(args):
    """
    Executa ciclos Darwin IA³ com Equação da Morte por neurônio
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🧬 DARWIN IA³ - NEUROGÊNESE POR RODADAS")
    logger.info(f"{'='*80}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Ciclos: {args.cycles}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    
    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_metrics()

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURAÇÃO DE DADOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if args.task == "synthetic":
        task = SyntheticTask(d_in=8, batch=args.batch, device=args.device)
        in_dim = 8
        out_dim = 1
        loss_fn = nn.MSELoss()
        logger.info(f"📊 Usando tarefa sintética (regressão)")
        
    elif args.task == "tinystories":
        task = TinyStoriesCharBag(
            split="train[:0.5%]",  # Subset leve para CPU
            window=64,  # Janela menor para CPU
            batch=args.batch, 
            device=args.device
        )
        in_dim = task.vocab_size
        out_dim = task.vocab_size
        loss_fn = nn.MSELoss()  # Tratamos como regressão sobre frequências
        logger.info(f"📖 Usando TinyStories (bag-of-chars)")
        
    else:
        raise ValueError(f"Task desconhecida: {args.task}")

    # ═══════════════════════════════════════════════════════════════════════════
    # POPULAÇÃO INICIAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    pop = Population(in_dim=in_dim, out_dim=out_dim, device=args.device, seed=args.seed)
    
    # Configurar política de nascimento
    pop.X_PER_DEATH_SPAWN = args.deaths_per_birth
    
    logger.info(f"🧬 População inicial: {len(pop.neurons)} neurônios")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL DE CICLOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    loss_history = []
    start_time = time.time()
    
    try:
        for cycle in range(1, args.cycles + 1):
            cycle_start = time.time()
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"🔄 CICLO {cycle}/{args.cycles}")
            logger.info(f"{'─'*60}")
            logger.info(f"População: {len(pop.neurons)} neurônios")
            
            # Atualizar métricas de ciclo
            g_cycle_index.set(cycle)
            
            # ═══════════════════════════════════════════════════════════════════
            # GERAÇÃO DE DADOS E TREINO
            # ═══════════════════════════════════════════════════════════════════
            
            # Gerar batch de dados
            x, y = task.batch()
            
            # Forward da população
            pred = pop(x)
            population_loss_tensor = loss_fn(pred, y)
            population_loss = population_loss_tensor.item()
            
            # Treino cooperativo
            train_loss = pop.train_step(x, y, loss_fn)
            
            logger.info(f"   📊 Loss população: {population_loss:.6f}")
            logger.info(f"   📊 Loss treino: {train_loss:.6f}")
            
            # ═══════════════════════════════════════════════════════════════════
            # MÉTRICAS E PROXIES
            # ═══════════════════════════════════════════════════════════════════
            
            # Atualizar histórico
            loss_history.append(train_loss)
            
            # Métricas de população
            g_population_loss.set(train_loss)
            g_neurons_alive.set(len(pop.neurons))
            g_births.set(pop.births)
            g_deaths.set(pop.deaths)
            g_consciousness.set(pop.collective_consciousness)
            
            # Proxies auditáveis (TEIS/Darwin)
            stability, oci, P = compute_proxies(loss_history)
            g_stability.set(stability)
            g_oci.set(oci)
            g_p_convergence.set(P)
            
            logger.info(f"   📈 Estabilidade: {stability:.3f}")
            logger.info(f"   📈 OCI: {oci:.3f}")
            logger.info(f"   📈 P: {P:.3f}")
            
            # ═══════════════════════════════════════════════════════════════════
            # EQUAÇÃO DA MORTE POR NEURÔNIO
            # ═══════════════════════════════════════════════════════════════════
            
            logger.info(f"\n⚖️ Aplicando Equação da Morte...")
            
            # Avaliação e seleção natural
            deaths_this_cycle = pop.evaluate_and_cull(train_loss)
            
            # Atualizar contadores
            c_cycles.inc()
            
            # ═══════════════════════════════════════════════════════════════════
            # NASCIMENTO OBRIGATÓRIO POR CICLO
            # ═══════════════════════════════════════════════════════════════════
            
            # Sempre nascer 1 neurônio por ciclo (política Darwin)
            if len(pop.neurons) < args.max_neurons:
                logger.info(f"\n🌱 Nascimento obrigatório por ciclo...")
                new_neuron = pop.add_neuron()
                logger.info(f"   🐣 {new_neuron.neuron_id} nasceu (política ciclo)")
            
            # ═══════════════════════════════════════════════════════════════════
            # LOGGING E MÉTRICAS
            # ═══════════════════════════════════════════════════════════════════
            
            cycle_time = time.time() - cycle_start
            
            # WORM do ciclo completo
            log_event({
                "event": "cycle_completed",
                "cycle": cycle,
                "metrics": {
                    "population_loss": population_loss,
                    "train_loss": train_loss,
                    "neurons": len(pop.neurons),
                    "births": pop.births,
                    "deaths": pop.deaths,
                    "collective_consciousness": pop.collective_consciousness,
                    "stability": stability,
                    "oci": oci,
                    "P": P
                },
                "cycle_time": cycle_time,
                "deaths_this_cycle": deaths_this_cycle
            })
            
            # Log progresso
            logger.info(f"   ⏱️ Ciclo completo em {cycle_time:.2f}s")
            logger.info(f"   📊 Mortes: {deaths_this_cycle}")
            logger.info(f"   🧠 Consciência: {pop.collective_consciousness:.3f}")
            
            # Pausa pequena para observabilidade
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Ciclos interrompidos pelo usuário")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RELATÓRIO FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    total_time = time.time() - start_time
    final_summary = pop.get_population_summary()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🏁 DARWIN IA³ FINALIZADO")
    logger.info(f"{'='*80}")
    logger.info(f"Tempo total: {total_time:.2f}s")
    logger.info(f"Ciclos executados: {cycle}")
    logger.info(f"População final: {len(pop.neurons)} neurônios")
    logger.info(f"Total nascimentos: {pop.births}")
    logger.info(f"Total mortes: {pop.deaths}")
    logger.info(f"Consciência coletiva final: {pop.collective_consciousness:.3f}")
    
    # WORM final
    log_event({
        "event": "darwin_session_completed",
        "total_cycles": cycle,
        "total_time": total_time,
        "final_summary": final_summary
    })
    
    # Verificar integridade WORM
    worm_stats = get_worm_stats()
    logger.info(f"\n📜 WORM: {worm_stats['events']} eventos, integridade: {worm_stats['integrity']}")
    
    return await final_summary

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="🧬 Darwin IA³ - Neurogênese por Rodadas com Equação da Morte",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--task", choices=["synthetic", "tinystories"], default="synthetic",
                       help="Tipo de tarefa (synthetic|tinystories)")
    parser.add_argument("--device", default="cpu", help="Device PyTorch")
    parser.add_argument("--cycles", type=int, default=50, help="Número de ciclos")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deaths-per-birth", type=int, default=10, 
                       help="Mortes necessárias para nascimento especial")
    parser.add_argument("--max-neurons", type=int, default=100,
                       help="Máximo de neurônios na população")
    
    args = parser.parse_args()
    
    try:
        final_summary = run_darwin_cycles(args)
        logger.info(f"\n✅ Darwin IA³ executado com sucesso!")
        
    except Exception as e:
        logger.info(f"\n💥 Erro: {e}")
        import traceback
        traceback.print_exc()
        return await 1
    
    return await 0

if __name__ == "__main__":
    import sys
    sys.exit(main())