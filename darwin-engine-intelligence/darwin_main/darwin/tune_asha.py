"""
HPO com Ray Tune + ASHA: explora lr, batch e política (X_PER_DEATH_SPAWN).
Executa ciclos curtos e seleciona configs promissoras.

Referência: Ray Tune + ASHA Scheduler
https://docs.ray.io/en/latest/tune/api/schedulers.html#asha-scheduler
"""
import os, random, time
import torch, torch.nn as nn

try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .data_sources import SyntheticTask
from .population import Population

async def objective(config):
    """
    Objetivo para otimização ASHA:
    Executa Darwin com configuração específica e retorna métricas
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray Tune não disponível. Instale com: pip install 'ray[tune]'")
    
    # Setup determinístico
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    # Task e população conforme config
    task = SyntheticTask(d_in=8, batch=config["batch"])
    loss_fn = nn.MSELoss()
    
    pop = Population(in_dim=8, out_dim=1, seed=config["seed"])
    pop.X_PER_DEATH_SPAWN = config["x_per_death_spawn"]
    
    # Ajustar learning rates da população
    for neuron in pop.neurons:
        for group in neuron.opt.param_groups:
            group["lr"] = config["lr"]
        for group in neuron.opt_arch.param_groups:
            group["lr"] = config["arch_lr"]
    
    # Executar ciclos curtos para avaliação
    losses = []
    consciousness_scores = []
    survival_rates = []
    
    budget_cycles = config.get("budget_cycles", 20)  # Orçamento curto para ASHA
    
    for cycle in range(budget_cycles):
        # Gerar dados
        x, y = task.batch()
        
        # Treino da população
        train_loss = pop.train_step(x, y, loss_fn)
        losses.append(train_loss)
        
        # Avaliação IA³ e seleção natural
        pop.evaluate_and_cull(train_loss)
        
        # Métricas para ASHA
        consciousness_scores.append(pop.collective_consciousness)
        
        # Taxa de sobrevivência (neurônios que passaram na avaliação IA³)
        if pop.neurons:
            passed = sum(1 for n in pop.neurons if n.stats.get("ia3_score", 0) >= 0.6)
            survival_rate = passed / len(pop.neurons)
        else:
            survival_rate = 0.0
        survival_rates.append(survival_rate)
        
        # Nascimento obrigatório se população muito pequena
        if len(pop.neurons) < 2:
            pop.add_neuron()
        
        # Reportar progresso intermediário para ASHA
        if cycle % 5 == 0:
            tune.report(
                intermediate_loss=train_loss,
                intermediate_consciousness=pop.collective_consciousness,
                intermediate_survival_rate=survival_rate,
                population_size=len(pop.neurons)
            )
    
    # Métricas finais
    final_loss = sum(losses[-5:]) / 5 if len(losses) >= 5 else (losses[-1] if losses else float("inf"))
    final_consciousness = sum(consciousness_scores[-5:]) / 5 if len(consciousness_scores) >= 5 else 0.0
    final_survival_rate = sum(survival_rates[-5:]) / 5 if len(survival_rates) >= 5 else 0.0
    
    # Função objetivo composta (minimizar)
    # Menor loss + maior consciência + maior taxa de sobrevivência
    objective_score = final_loss - 0.5 * final_consciousness - 0.3 * final_survival_rate
    
    # Reportar resultado final
    tune.report(
        final_loss=final_loss,
        final_consciousness=final_consciousness,
        final_survival_rate=final_survival_rate,
        objective_score=objective_score,
        final_neurons=len(pop.neurons),
        total_births=pop.births,
        total_deaths=pop.deaths
    )

async def run_asha_optimization():
    """
    Executa otimização ASHA para encontrar melhores hiperparâmetros Darwin
    """
    if not RAY_AVAILABLE:
        logger.info("⚠️ Ray Tune não disponível. Instale com: pip install 'ray[tune]'")
        return await None
    
    logger.info("🔍 Iniciando otimização HPO com Ray Tune + ASHA...")
    
    # Espaço de busca
    search_space = {
        "seed": 42,  # Fixo para reprodutibilidade
        "batch": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "arch_lr": tune.loguniform(1e-5, 1e-3),
        "x_per_death_spawn": tune.choice([5, 10, 15, 20]),
        "budget_cycles": tune.choice([15, 20, 25])
    }
    
    # Scheduler ASHA (early stopping agressivo)
    scheduler = ASHAScheduler(
        metric="objective_score",
        mode="min",
        max_t=25,  # Máximo de ciclos por trial
        grace_period=5,  # Mínimo antes de parar
        reduction_factor=2  # Agressividade do corte
    )
    
    # Configuração do Tuner
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="objective_score",
            mode="min",
            scheduler=scheduler,
            num_samples=20  # Número de configurações a testar
        ),
        run_config=tune.RunConfig(
            name="darwin_ia3_asha",
            local_dir="/root/darwin/logs/ray_results"
        ),
        param_space=search_space,
    )
    
    # Executar otimização
    logger.info(f"   🎯 Testando {search_space} configurações...")
    logger.info(f"   ⚡ ASHA: early stopping agressivo ativo")
    
    results = tuner.fit()
    
    # Melhor resultado
    best_result = results.get_best_result()
    best_config = best_result.config
    best_metrics = best_result.metrics
    
    logger.info(f"\n🏆 MELHOR CONFIGURAÇÃO ENCONTRADA:")
    logger.info(f"   LR: {best_config['lr']:.6f}")
    logger.info(f"   Arch LR: {best_config['arch_lr']:.6f}")
    logger.info(f"   Batch: {best_config['batch']}")
    logger.info(f"   Mortes por nascimento: {best_config['x_per_death_spawn']}")
    logger.info(f"   Loss final: {best_metrics['final_loss']:.6f}")
    logger.info(f"   Consciência final: {best_metrics['final_consciousness']:.3f}")
    logger.info(f"   Taxa sobrevivência: {best_metrics['final_survival_rate']:.3f}")
    
    # Salvar melhor configuração
    config_path = "/root/darwin/logs/best_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        import json
        json.dump({
            "best_config": best_config,
            "best_metrics": best_metrics,
            "optimization_time": time.time(),
            "total_trials": len(results)
        }, f, indent=2)
    
    logger.info(f"   💾 Configuração salva em: {config_path}")
    
    return await best_config, best_metrics

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--optimize":
        # Executar otimização ASHA
        if RAY_AVAILABLE:
            run_asha_optimization()
        else:
            logger.info("❌ Ray Tune não disponível")
    else:
        logger.info("Uso:")
        logger.info("  python tune_asha.py --optimize    # Executar otimização ASHA")
        logger.info("")
        logger.info("Requires: pip install 'ray[tune]'")
        logger.info("Docs: https://docs.ray.io/en/latest/tune/api/schedulers.html#asha-scheduler")