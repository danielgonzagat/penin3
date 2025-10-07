#!/usr/bin/env python3
# darwinctl.py — CLI Darwin v3: spawn/round/rollback + hooks declarativos
import argparse, json, os, sys, time, hashlib, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Adicionar path local
sys.path.append(str(Path(__file__).resolve().parent))

from neurogenesis import Brain, MicroNeuron
from ia3_checks import IA3Inspector
from wormlog import WormLog
from prom_metrics import ensure_metrics_server, METRICS, GAUGES, observe_neuron_birth, observe_neuron_death, observe_round_metrics

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(os.getenv("DARWIN_DATA", "/root/darwin_data"))
SNAP_DIR = DATA_DIR / "snapshots"
WORM_LOG_PATH = DATA_DIR / "ia3_worm.log"
BRAIN_SNAPSHOT = SNAP_DIR / "brain_state.pt"
CONFIG_PATH = Path(os.getenv("DARWIN_CONFIG", "/root/darwin/config.json"))
HOOKS_DIR = Path("/root/darwin/hooks.d")

# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE HOOKS DECLARATIVO
# ═══════════════════════════════════════════════════════════════════════════════

async def run_hooks(event_type: str, env_vars: Dict[str, str]) -> Dict[str, Any]:
    """Executa hooks de um tipo específico em ordem"""
    hooks_path = HOOKS_DIR / event_type
    
    if not hooks_path.exists():
        return await {"executed": [], "failed": [], "total": 0}
    
    # Encontrar hooks executáveis
    hooks = sorted([
        h for h in hooks_path.iterdir() 
        if h.is_file() and os.access(h, os.X_OK)
    ])
    
    results = {"executed": [], "failed": [], "total": len(hooks)}
    
    if not hooks:
        return await results
    
    logger.info(f"🔧 Executando {len(hooks)} hooks {event_type}:")
    
    # Preparar ambiente
    hook_env = {**os.environ, **env_vars}
    
    for hook in hooks:
        try:
            logger.info(f"   ▶️ {hook.name}...")
            
            start_time = time.time()
            result = subprocess.run(
                [str(hook)],
                env=hook_env,
                capture_output=True,
                text=True,
                timeout=30
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"      ✅ Sucesso ({duration:.2f}s)")
                results["executed"].append({
                    "hook": hook.name,
                    "duration": duration,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip()
                })
            else:
                logger.info(f"      ❌ Falha (código {result.returncode})")
                results["failed"].append({
                    "hook": hook.name,
                    "returncode": result.returncode,
                    "duration": duration,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip()
                })
                
                # Código 3 = neurônio reprovado na avaliação IA³
                if result.returncode == 3:
                    logger.info(f"      ☠️ Neurônio reprovado na avaliação IA³")
        
        except subprocess.TimeoutExpired:
            logger.info(f"      ⏰ Timeout ({hook.name})")
            results["failed"].append({
                "hook": hook.name,
                "error": "timeout",
                "duration": 30.0
            })
        except Exception as e:
            logger.info(f"      💥 Erro ({hook.name}): {e}")
            results["failed"].append({
                "hook": hook.name,
                "error": str(e),
                "duration": 0.0
            })
    
    return await results

# ═══════════════════════════════════════════════════════════════════════════════
# HOOKS CORE (neurogenesis primitives)
# ═══════════════════════════════════════════════════════════════════════════════

async def hook_spawn(brain: Brain, reason: str = "cycle_end_birth") -> Dict[str, Any]:
    """Nasce 1 neurônio (neurogênese Net2Wider), conecta-se ao grafo"""
    old_count = len(brain.neurons)
    neuron_id = brain.add_neuron()  # Usa Net2Wider se possível
    new_count = len(brain.neurons)
    
    observe_neuron_birth(neuron_id, new_count)
    
    return await {
        "event": "spawn",
        "neuron_id": neuron_id,
        "reason": reason,
        "neurons_before": old_count,
        "neurons_after": new_count
    }

async def hook_kill(brain: Brain, neuron_id: str, reason: str, details: dict) -> Dict[str, Any]:
    """Mata 1 neurônio não IA3-like"""
    old_count = len(brain.neurons)
    removed = brain.remove_neuron(neuron_id)
    new_count = len(brain.neurons)
    
    if removed:
        observe_neuron_death(neuron_id, new_count, details.get("ia3_score", 0.0))
    
    return await {
        "event": "kill",
        "neuron_id": neuron_id,
        "reason": reason,
        "removed": removed,
        "neurons_before": old_count,
        "neurons_after": new_count,
        "details": details
    }

async def hook_rollback(brain: Brain, to_path: Path) -> Dict[str, Any]:
    """Restaura snapshot completo do cérebro"""
    old_generation = brain.generation
    old_neurons = len(brain.neurons)
    
    # Carregar estado anterior
    if to_path.exists():
        restored_brain = Brain.load(to_path)
        brain.load_state_dict(restored_brain.state_dict())
        brain.generation = restored_brain.generation
        brain.neurons = restored_brain.neurons
    
    METRICS["rollbacks"].inc()
    
    return await {
        "event": "rollback",
        "snapshot": str(to_path),
        "generation_before": old_generation,
        "generation_after": brain.generation,
        "neurons_before": old_neurons,
        "neurons_after": len(brain.neurons)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════════

async def ensure_dirs():
    """Garante que diretórios existem"""
    for path in (DATA_DIR, SNAP_DIR, HOOKS_DIR):
        path.mkdir(parents=True, exist_ok=True)

async def load_config() -> Dict[str, Any]:
    """Carrega configuração Darwin"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return await json.load(f)
    
    # Configuração padrão
    default_config = {
        "birth_every_round": True,
        "ia3_threshold": 0.60,
        "delta_loss_min": 0.001,
        "max_neurons": 4096,
        "budget_sec_per_round": 60,
        "seed": 42,
        "deaths_for_bonus_birth": 10
    }
    
    # Salvar configuração padrão
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    return await default_config

async def create_snapshot(brain: Brain, path: Path) -> Dict[str, Any]:
    """Cria snapshot do cérebro com hash SHA-256"""
    brain.save(path)
    
    # Calcular hash do arquivo
    with open(path, 'rb') as f:
        content = f.read()
    sha256 = hashlib.sha256(content).hexdigest()
    
    return await {
        "snapshot": str(path),
        "sha256": sha256,
        "size_bytes": len(content),
        "neurons": len(brain.neurons),
        "generation": brain.generation
    }

# ═══════════════════════════════════════════════════════════════════════════════
# COMANDOS CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def cmd_init(args):
    """Inicializa sistema Darwin v3"""
    ensure_dirs()
    
    # Iniciar servidor de métricas
    port = ensure_metrics_server(port=args.metrics_port)
    
    # Log inicial no WORM
    worm = WormLog(WORM_LOG_PATH)
    worm.append({
        "event": "darwin_init",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "v3_advanced",
        "metrics_port": port
    })
    
    logger.info(f"✅ Darwin v3 inicializado")
    logger.info(f"   Métricas: :{port}")
    logger.info(f"   WORM: {WORM_LOG_PATH}")
    logger.info(f"   Hooks: {HOOKS_DIR}")

async def cmd_spawn(args):
    """Força spawn de neurônio"""
    ensure_dirs()
    
    brain = Brain.load_or_new(BRAIN_SNAPSHOT)
    
    # Preparar ambiente para hooks
    env_vars = {
        "EVENT": "spawn",
        "NEURON_ID": f"manual_{int(time.time())}",
        "REASON": args.reason,
        "TIMESTAMP": datetime.utcnow().isoformat() + "Z"
    }
    
    # PRE-spawn hooks
    pre_results = run_hooks("pre_spawn", env_vars)
    
    # Spawn principal
    spawn_result = hook_spawn(brain, reason=args.reason)
    
    # POST-spawn hooks
    env_vars["NEURON_ID"] = spawn_result["neuron_id"]
    post_results = run_hooks("post_spawn", env_vars)
    
    # Snapshot
    snapshot_result = create_snapshot(brain, BRAIN_SNAPSHOT)
    
    # WORM log
    worm = WormLog(WORM_LOG_PATH)
    worm.append({
        "timestamp": env_vars["TIMESTAMP"],
        **spawn_result,
        "pre_hooks": pre_results,
        "post_hooks": post_results,
        **snapshot_result
    })
    
    # Output
    result = {
        **spawn_result,
        "pre_hooks": pre_results,
        "post_hooks": post_results,
        **snapshot_result
    }
    
    logger.info(json.dumps(result, indent=2))

async def cmd_round(args):
    """Executa round completo: treino + julgamento IA³ + morte/nascimento"""
    ensure_dirs()
    
    cfg = load_config()
    brain = Brain.load_or_new(BRAIN_SNAPSHOT)
    inspector = IA3Inspector(cfg, brain)
    
    round_start = time.time()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔄 ROUND DARWIN v3")
    logger.info(f"{'='*80}")
    logger.info(f"Neurônios ativos: {len(brain.neurons)}")
    logger.info(f"Geração: {brain.generation}")
    logger.info(f"Budget: {cfg['budget_sec_per_round']}s")
    
    # 1. Treino leve + avaliação do round
    logger.info(f"\n📚 Executando treino adaptativo...")
    round_metrics = brain.run_round(budget_sec=cfg["budget_sec_per_round"])
    round_metrics["timestamp"] = timestamp
    
    logger.info(f"   ✅ Treino concluído: {round_metrics['steps']} steps, loss {round_metrics['avg_loss']:.6f}")
    
    # 2. Julgamento IA³ por neurônio + Equação da Morte
    logger.info(f"\n⚖️ Aplicando julgamento IA³ + Equação da Morte...")
    verdicts = inspector.judge_all_neurons(round_metrics)
    
    # 2.1 Registrar neurônios IA³ validados (passaram) no WORM dedicado
    try:
        worm_ia3 = WormLog(WORM_LOG_PATH)
        for v in verdicts:
            if v.get("passes"):
                worm_ia3.append({
                    "event": "ia3_neuron_validated",
                    "timestamp": v.get("timestamp"),
                    "neuron_id": v.get("neuron_id"),
                    "round_number": v.get("round_number"),
                    "ia3_score": v.get("ia3_score"),
                    "criteria_passed": {k: bool(v.get("criteria", {}).get(k)) for k in (v.get("criteria", {}) or {}).keys()}
                })
    except Exception:
        # Não bloquear o round em caso de falha de registro
        pass

    # Identificar neurônios para execução
    death_sentences = [(v["neuron_id"], v) for v in verdicts if not v["passes"]]
    
    # 3. Executar Equação da Morte
    if death_sentences:
        logger.info(f"\n☠️ Executando {len(death_sentences)} neurônios reprovados...")
        
        for neuron_id, verdict in death_sentences:
            kill_result = hook_kill(brain, neuron_id, reason="ia3_evaluation_failed", details=verdict)
            
            # Log no WORM
            worm = WormLog(WORM_LOG_PATH)
            worm.append({
                "timestamp": timestamp,
                **kill_result
            })
    
    # 4. Política de nascimento
    births = []
    
    # Nascimento obrigatório por rodada
    if cfg["birth_every_round"] and len(brain) < cfg["max_neurons"]:
        spawn_result = hook_spawn(brain, reason="policy_birth_every_round")
        births.append(spawn_result)
        logger.info(f"🐣 Nascimento obrigatório: {spawn_result['neuron_id']}")
    
    # Nascimento por contador de mortes
    death_counter = METRICS["neurons_killed"]._value.get() if hasattr(METRICS["neurons_killed"], '_value') else brain.total_deaths
    births_per_deaths = cfg.get("deaths_for_bonus_birth", 10)
    
    if death_counter > 0 and death_counter % births_per_deaths == 0:
        spawn_result = hook_spawn(brain, reason=f"death_counter_birth_{births_per_deaths}")
        births.append(spawn_result)
        logger.info(f"🎁 Nascimento bonus: {spawn_result['neuron_id']} ({death_counter} mortes)")
    
    # 5. Verificar extinção total
    if len(brain.neurons) == 0:
        logger.info(f"💀 EXTINÇÃO TOTAL detectada!")
        
        # Renascer mínimo
        brain.generation += 1
        neuron_id = brain.add_neuron(act="relu")
        
        extinction_event = {
            "event": "total_extinction_and_rebirth",
            "timestamp": timestamp,
            "new_generation": brain.generation,
            "reborn_neuron": neuron_id
        }
        
        worm = WormLog(WORM_LOG_PATH)
        worm.append(extinction_event)
        
        logger.info(f"🌱 Renascimento: nova geração {brain.generation}, neurônio {neuron_id}")
    
    # 6. Snapshot final
    snapshot_result = create_snapshot(brain, BRAIN_SNAPSHOT)
    
    # 7. Métricas e WORM final
    pass_rate = sum(1 for v in verdicts if v["passes"]) / len(verdicts) if verdicts else 0.0
    
    observe_round_metrics(
        avg_loss=round_metrics["avg_loss"],
        ia3_pass_rate=pass_rate,
        consciousness=round_metrics["collective_consciousness"],
        training_time=round_metrics["training_time"]
    )
    
    # WORM do round completo
    worm = WormLog(WORM_LOG_PATH)
    worm.append({
        "event": "round_complete",
        "timestamp": timestamp,
        "round_metrics": round_metrics,
        "verdicts_summary": {
            "total": len(verdicts),
            "passed": sum(1 for v in verdicts if v["passes"]),
            "failed": len(death_sentences),
            "pass_rate": pass_rate
        },
        "births": births,
        "deaths": len(death_sentences),
        **snapshot_result
    })
    
    # Output final
    round_time = time.time() - round_start
    
    result = {
        "round_duration": round_time,
        "round_metrics": round_metrics,
        "verdicts": len(verdicts),
        "passed": sum(1 for v in verdicts if v["passes"]),
        "executed": len(death_sentences),
        "births": len(births),
        "final_neurons": len(brain.neurons),
        "pass_rate": pass_rate,
        **snapshot_result
    }
    
    logger.info(f"\n✅ ROUND CONCLUÍDO ({round_time:.2f}s)")
    logger.info(f"   Neurônios finais: {len(brain.neurons)}")
    logger.info(f"   Taxa aprovação: {pass_rate*100:.1f}%")
    logger.info(f"   Execuções: {len(death_sentences)}")
    logger.info(f"   Nascimentos: {len(births)}")
    
    logger.info(json.dumps(result, indent=2))

async def cmd_rollback(args):
    """Executa rollback para snapshot anterior"""
    ensure_dirs()
    
    brain = Brain.load_or_new(BRAIN_SNAPSHOT)
    rollback_path = Path(args.snapshot)
    
    # Preparar ambiente para hooks
    env_vars = {
        "EVENT": "rollback",
        "REASON": args.reason,
        "TIMESTAMP": datetime.utcnow().isoformat() + "Z",
        "SNAPSHOT_PATH": str(rollback_path),
        "MODEL_SYMLINK": str(Path("/root/current_model"))
    }
    
    # PRE-rollback hooks
    pre_results = run_hooks("pre_rollback", env_vars)
    
    # Rollback principal
    rollback_result = hook_rollback(brain, rollback_path)
    
    # POST-rollback hooks
    post_results = run_hooks("post_rollback", env_vars)
    
    # Salvar estado restaurado
    brain.save(BRAIN_SNAPSHOT)
    
    # WORM log
    worm = WormLog(WORM_LOG_PATH)
    worm.append({
        "timestamp": env_vars["TIMESTAMP"],
        **rollback_result,
        "pre_hooks": pre_results,
        "post_hooks": post_results
    })
    
    result = {
        **rollback_result,
        "pre_hooks": pre_results,
        "post_hooks": post_results
    }
    
    logger.info(json.dumps(result, indent=2))

async def cmd_status(args):
    """Mostra status completo do sistema"""
    ensure_dirs()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧬 STATUS DARWIN v3 NEUROEVOLUTIVO")
    logger.info(f"{'='*60}")
    
    # Status do cérebro
    if BRAIN_SNAPSHOT.exists():
        brain = Brain.load(BRAIN_SNAPSHOT)
        logger.info(f"🧠 CÉREBRO:")
        logger.info(f"   Neurônios: {len(brain.neurons)}")
        logger.info(f"   Geração: {brain.generation}")
        logger.info(f"   Nascimentos: {brain.total_births}")
        logger.info(f"   Mortes: {brain.total_deaths}")
        
        if brain.neurons:
            logger.info(f"   Consciência média: {sum(n.consciousness_score for n in brain.neurons)/len(brain.neurons):.3f}")
            logger.info(f"   Idade média: {sum(n.age for n in brain.neurons)/len(brain.neurons):.1f}")
    else:
        logger.info(f"🧠 CÉREBRO: Não inicializado")
    
    # Status WORM
    logger.info(f"\n📜 WORM LOG:")
    if WORM_LOG_PATH.exists():
        try:
            worm = WormLog(WORM_LOG_PATH)
            is_valid, msg = worm.verify_chain()
            
            with open(WORM_LOG_PATH, 'r') as f:
                lines = f.readlines()
            
            events = [l for l in lines if not l.startswith("HASH:")]
            logger.info(f"   Eventos: {len(events)}")
            logger.info(f"   Integridade: {'✅' if is_valid else '❌'} ({msg})")
            
            if events:
                try:
                    last_event = json.loads(events[-1])
                    logger.info(f"   Último: {last_event.get('event', 'unknown')} ({last_event.get('timestamp', 'unknown')})")
                except:
                    pass
        except Exception as e:
            logger.info(f"   ⚠️ Erro: {e}")
    else:
        logger.info(f"   Arquivo não existe: {WORM_LOG_PATH}")
    
    # Status dos hooks
    logger.info(f"\n🔧 HOOKS:")
    for event_type in ["pre_spawn", "post_spawn", "pre_rollback", "post_rollback"]:
        event_dir = HOOKS_DIR / event_type
        if event_dir.exists():
            hooks = [h for h in event_dir.iterdir() if h.is_file() and os.access(h, os.X_OK)]
            logger.info(f"   {event_type}: {len(hooks)} hooks")
        else:
            logger.info(f"   {event_type}: ❌ Não existe")

async def cmd_create_hooks(args):
    """Cria hooks de exemplo"""
    logger.info(f"🔧 Criando hooks de exemplo Darwin v3...")
    
    ensure_dirs()
    
    # Hook post-spawn: log básico
    post_spawn_dir = HOOKS_DIR / "post_spawn"
    post_spawn_dir.mkdir(exist_ok=True)
    
    hook_log = post_spawn_dir / "20-log-neuron.sh"
    with open(hook_log, 'w') as f:
        f.write("""#!/usr/bin/env bash
set -euo pipefail
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SPAWN: id=$NEURON_ID reason=$REASON" >> /root/ia3_darwin_hooks.log
echo "✅ Neurônio $NEURON_ID logado"
""")
    os.chmod(hook_log, 0o755)
    
    # Hook post-spawn: avaliação IA³ rigorosa
    hook_eval = post_spawn_dir / "30-evaluate-ia3.py"
    with open(hook_eval, 'w') as f:
        f.write("""#!/usr/bin/env python3
import os, json, sys, time, random
sys.path.append("/root/darwin")

try:
    from neurogenesis import Brain
    from ia3_checks import IA3Inspector
    
    # Simular avaliação IA³ (em hook real, usar dados reais)
    neuron_id = os.getenv("NEURON_ID", "unknown")
    logger.info(f"🔬 Avaliando {neuron_id} contra critérios IA³...")
    
    # Critérios simulados (em produção, usar avaliação real)
    ia3_score = random.uniform(0.3, 0.9)
    consciousness = random.uniform(0.1, 0.8)
    
    passes = ia3_score >= 0.60 and consciousness >= 0.3
    
    result = {
        "neuron_id": neuron_id,
        "ia3_score": ia3_score,
        "consciousness": consciousness,
        "passes": passes,
        "timestamp": os.getenv("TIMESTAMP")
    }
    
    logger.info(json.dumps(result, indent=2))
    
    if not passes:
        logger.info(f"❌ Neurônio {neuron_id} REPROVADO (score: {ia3_score:.3f})")
        sys.exit(3)  # Código especial para falha IA³
    else:
        logger.info(f"✅ Neurônio {neuron_id} APROVADO (score: {ia3_score:.3f})")

except Exception as e:
    logger.info(f"💥 Erro: {e}")
    sys.exit(1)
""")
    os.chmod(hook_eval, 0o755)
    
    # Hook post-rollback: notificação
    post_rollback_dir = HOOKS_DIR / "post_rollback" 
    post_rollback_dir.mkdir(exist_ok=True)
    
    hook_notify = post_rollback_dir / "10-notify.sh"
    with open(hook_notify, 'w') as f:
        f.write("""#!/usr/bin/env bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ROLLBACK: reason=$REASON success=${ROLLBACK_SUCCESS:-0}" >> /root/ia3_darwin_hooks.log
echo "✅ Rollback notificado: $REASON"
""")
    os.chmod(hook_notify, 0o755)
    
    logger.info(f"✅ Hooks de exemplo criados:")
    logger.info(f"   {hook_log}")
    logger.info(f"   {hook_eval}")
    logger.info(f"   {hook_notify}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🧬 Darwin v3 CLI - Sistema Neuroevolutivo Avançado",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Init
    init_parser = subparsers.add_parser('init', help='Inicializa sistema Darwin v3')
    init_parser.add_argument('--metrics-port', type=int, default=9092, help='Porta métricas')
    init_parser.set_defaults(func=cmd_init)
    
    # Spawn
    spawn_parser = subparsers.add_parser('spawn', help='Força spawn de neurônio')
    spawn_parser.add_argument('--reason', default='manual', help='Razão do spawn')
    spawn_parser.set_defaults(func=cmd_spawn)
    
    # Round
    round_parser = subparsers.add_parser('round', help='Executa round completo')
    round_parser.set_defaults(func=cmd_round)
    
    # Rollback
    rollback_parser = subparsers.add_parser('rollback', help='Rollback para snapshot')
    rollback_parser.add_argument('--snapshot', required=True, help='Caminho do snapshot')
    rollback_parser.add_argument('--reason', default='manual', help='Razão do rollback')
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # Status
    status_parser = subparsers.add_parser('status', help='Status do sistema')
    status_parser.set_defaults(func=cmd_status)
    
    # Create hooks
    hooks_parser = subparsers.add_parser('create-hooks', help='Criar hooks de exemplo')
    hooks_parser.set_defaults(func=cmd_create_hooks)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            logger.info("\n🛑 Comando interrompido")
            sys.exit(1)
        except Exception as e:
            logger.info(f"💥 Erro: {e}")
            sys.exit(1)
    else:
        parser.print_help()