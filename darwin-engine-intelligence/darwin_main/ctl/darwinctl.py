#!/usr/bin/env python3
# /root/darwin/ctl/darwinctl.py
# Controlador Darwin v4: spawn/round/rollback + hooks + WORM + contador de mortes=10 → nascimento

import os, json, hashlib, time, signal, sys, shutil, argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from pathlib import Path

# Adicionar paths do sistema Darwin
sys.path.append("/root/darwin")
sys.path.append("/root/darwin/neurons")
sys.path.append("/root/darwin/rules")
sys.path.append("/root/darwin/metrics")

from neurons.ia3_neuron import IA3Neuron, TrainConfig
from rules.equacao_da_morte import equacao_da_morte, DeathEquationConfig, calculate_heritage

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO E PATHS
# ═══════════════════════════════════════════════════════════════════════════════

WORM = "/root/darwin/logs/worm.log"
STATE = "/root/darwin/logs/state.json"
CHECKPOINTS = "/root/darwin/checkpoints"
SURVIVORS_LOG = "/root/darwin/logs/survivors.log"
DEATHS_LOG = "/root/darwin/logs/deaths.log"

# ═══════════════════════════════════════════════════════════════════════════════
# WORM LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

async def _worm_append(event: Dict[str, Any]):
    """Adiciona evento ao WORM log com hash chain"""
    os.makedirs(os.path.dirname(WORM), exist_ok=True)
    
    # Calcular hash da cadeia anterior
    prev_hash = "GENESIS"
    if os.path.exists(WORM):
        with open(WORM, "rb") as f:
            content = f.read()
            if content:
                prev_hash = hashlib.sha256(content).hexdigest()[:16]
    
    # Criar evento completo
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "previous_hash": prev_hash,
        **event
    }
    
    # Calcular hash do evento atual
    event_json = json.dumps(payload, sort_keys=True)
    current_hash = hashlib.sha256(event_json.encode()).hexdigest()[:16]
    
    # Escrever no WORM
    with open(WORM, "a") as f:
        f.write(f"EVENT:{event_json}\n")
        f.write(f"HASH:{current_hash}\n")
    
    return await current_hash

async def _load_state() -> Dict[str, Any]:
    """Carrega estado persistente do Darwin"""
    if os.path.exists(STATE):
        with open(STATE) as f:
            return await json.load(f)
    
    return await {
        "deaths": 0,
        "births": 0,
        "survivals": 0,
        "rounds": 0,
        "death_streak": 0,
        "best_val": None,
        "best_cfg": None,
        "last_neuron_id": None,
        "current_generation": 0
    }

async def _save_state(state: Dict[str, Any]):
    """Salva estado persistente do Darwin"""
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# HOOKS INTEGRADOS
# ═══════════════════════════════════════════════════════════════════════════════

async def hook_spawn(cfg: TrainConfig, reason: str = "cycle_birth") -> Tuple[IA3Neuron, str]:
    """
    Hook de spawn: cria neurônio IA³ e registra no WORM
    """
    logger.info(f"🐣 HOOK SPAWN: Criando neurônio IA³...")
    
    neuron = IA3Neuron(cfg)
    
    # Log no WORM
    spawn_event = {
        "event": "neuron_spawn",
        "neuron_id": neuron.neuron_id,
        "reason": reason,
        "config": asdict(cfg),
        "generation": neuron.generation
    }
    
    hash_id = _worm_append(spawn_event)
    
    logger.info(f"   ✅ Neurônio {neuron.neuron_id} criado")
    logger.info(f"   📝 WORM hash: {hash_id}")
    
    return await neuron, hash_id

async def hook_kill(neuron_id: str, reason: str, metrics: Dict[str, Any]) -> str:
    """
    Hook de kill: mata neurônio e registra no WORM
    """
    logger.info(f"☠️ HOOK KILL: Executando neurônio {neuron_id}...")
    
    kill_event = {
        "event": "neuron_death",
        "neuron_id": neuron_id,
        "reason": reason,
        "final_metrics": metrics,
        "execution_time": datetime.utcnow().isoformat() + "Z"
    }
    
    hash_id = _worm_append(kill_event)
    
    # Log no arquivo de mortes
    os.makedirs(os.path.dirname(DEATHS_LOG), exist_ok=True)
    with open(DEATHS_LOG, "a") as f:
        f.write(json.dumps(kill_event, ensure_ascii=False) + "\n")
    
    logger.info(f"   💀 Neurônio {neuron_id} executado")
    logger.info(f"   📝 WORM hash: {hash_id}")
    logger.info(f"   📊 Razão: {reason}")
    
    return await hash_id

async def hook_promote(neuron_id: str, metrics: Dict[str, Any]) -> str:
    """
    Hook de promoção: neurônio sobreviveu e é promovido
    """
    logger.info(f"✅ HOOK PROMOTE: Promovendo neurônio {neuron_id}...")
    
    promote_event = {
        "event": "neuron_promotion",
        "neuron_id": neuron_id,
        "survival_metrics": metrics,
        "promotion_time": datetime.utcnow().isoformat() + "Z"
    }
    
    hash_id = _worm_append(promote_event)
    
    # Log no arquivo de sobreviventes
    os.makedirs(os.path.dirname(SURVIVORS_LOG), exist_ok=True)
    with open(SURVIVORS_LOG, "a") as f:
        f.write(json.dumps(promote_event, ensure_ascii=False) + "\n")
    
    logger.info(f"   🏆 Neurônio {neuron_id} promovido")
    logger.info(f"   📝 WORM hash: {hash_id}")
    logger.info(f"   📊 Score IA³: {metrics.get('ia3_score', 0):.3f}")
    
    return await hash_id

async def hook_rollback(snapshot_path: str) -> str:
    """
    Hook de rollback: restaura estado anterior
    """
    logger.info(f"🔄 HOOK ROLLBACK: Restaurando {snapshot_path}...")
    
    rollback_event = {
        "event": "system_rollback",
        "snapshot_path": snapshot_path,
        "rollback_time": datetime.utcnow().isoformat() + "Z",
        "reason": "manual_rollback"
    }
    
    hash_id = _worm_append(rollback_event)
    
    logger.info(f"   🔙 Rollback executado")
    logger.info(f"   📝 WORM hash: {hash_id}")
    
    return await hash_id

# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES PRINCIPAIS DO SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

async def create_snapshot(metrics: Dict[str, Any], neuron_state: Optional[Dict] = None) -> str:
    """Cria snapshot do estado atual"""
    os.makedirs(CHECKPOINTS, exist_ok=True)
    
    snapshot_name = f"snapshot_{int(time.time())}.json"
    snapshot_path = os.path.join(CHECKPOINTS, snapshot_name)
    
    snapshot_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "neuron_state": neuron_state,
        "system_state": _load_state()
    }
    
    with open(snapshot_path, "w") as f:
        json.dump(snapshot_data, f, indent=2)
    
    # Calcular hash
    with open(snapshot_path, "rb") as f:
        content = f.read()
    
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    
    logger.info(f"📸 Snapshot criado: {snapshot_name} (hash: {file_hash})")
    
    return await snapshot_path

async def one_round() -> Dict[str, Any]:
    """
    Executa UMA rodada completa Darwin v4:
    1. Carrega estado
    2. Spawn neurônio (com herança)
    3. Treina e avalia
    4. Aplica Equação da Morte
    5. Mata ou promove
    6. Nascimento a cada 10 mortes
    7. Snapshot e WORM
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🔄 ROUND DARWIN V4 - NEUROGÊNESE VIVA")
    logger.info(f"{'='*80}")
    
    # Carregar estado
    state = _load_state()
    state["rounds"] += 1
    
    logger.info(f"Round #{state['rounds']}")
    logger.info(f"Estado atual: {state['survivals']} vivos, {state['deaths']} mortes, {state['births']} nascimentos")
    
    # Atualizar métricas de round
    try:
        from metrics.metrics_server import update_round_start
        update_round_start(1)  # 1 neurônio por round
    except ImportError:
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HERANÇA E CONFIGURAÇÃO
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Carregar histórico para herança
    survivors_history = []
    deaths_history = []
    
    if os.path.exists(SURVIVORS_LOG):
        with open(SURVIVORS_LOG, 'r') as f:
            survivors_history = [json.loads(line) for line in f if line.strip()]
    
    if os.path.exists(DEATHS_LOG):
        with open(DEATHS_LOG, 'r') as f:
            deaths_history = [json.loads(line) for line in f if line.strip()]
    
    # Calcular herança
    heritage = calculate_heritage(survivors_history, deaths_history)
    
    # Configuração do neurônio (herança ou padrão)
    if state["best_cfg"] is None or heritage["strategy"] == "random_initialization":
        cfg = TrainConfig(
            seed=42 + state["rounds"],
            act="relu",
            lr=0.01,
            steps=200
        )
    else:
        # Usar configuração do melhor + variação
        best_cfg = state["best_cfg"]
        cfg = TrainConfig(
            input_dim=best_cfg.get("input_dim", 16),
            hidden_dim=best_cfg.get("hidden_dim", 16),
            lr=best_cfg.get("lr", 0.01) * (0.8 + random.random() * 0.4),  # Variação
            steps=best_cfg.get("steps", 200),
            seed=42 + state["rounds"],
            batch=best_cfg.get("batch", 64),
            device=best_cfg.get("device", "cpu"),
            act=random.choice(["relu", "gelu", "tanh"])  # Diversificar ativação
        )
    
    logger.info(f"🧬 Configuração: lr={cfg.lr:.4f}, act={cfg.act}, seed={cfg.seed}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPAWN E TREINAMENTO
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Spawn neurônio
    neuron, spawn_hash = hook_spawn(cfg, reason="round_cycle")
    
    # Diretório de trabalho para este round
    workdir = f"/root/darwin/logs/round_{state['rounds']:06d}_{neuron.neuron_id}"
    
    # Treinar e avaliar
    logger.info(f"🎯 Executando ciclo de vida completo...")
    metrics = neuron.fit_eval(workdir=workdir)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EQUAÇÃO DA MORTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    logger.info(f"\n⚖️ APLICANDO EQUAÇÃO DA MORTE...")
    death_cfg = DeathEquationConfig()
    death_result = equacao_da_morte(metrics, death_cfg)
    
    E = death_result["E"]
    decision = death_result["decision"]
    
    logger.info(f"   Decisão: {decision} (E={E})")
    logger.info(f"   A(t): {death_result['A']}, C(t): {death_result['C']}")
    logger.info(f"   Score IA³: {death_result['ia3_score']:.3f}")
    logger.info(f"   Consciência: {death_result['consciousness_score']:.3f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VIDA OU MORTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    if E == 1:
        # SOBREVIVEU - Promover
        state["survivals"] += 1
        state["death_streak"] = 0  # Reset streak de mortes
        
        # Atualizar best config se melhorou
        current_val = metrics["post_val_loss"]
        if state["best_val"] is None or current_val < state["best_val"]:
            state["best_val"] = current_val
            state["best_cfg"] = asdict(cfg)
            logger.info(f"   🏆 NOVO RECORDE! Val loss: {current_val:.6f}")
        
        # Hook de promoção
        promote_hash = hook_promote(neuron.neuron_id, {**metrics, **death_result})
        
        # Atualizar métricas Prometheus
        try:
            from metrics.metrics_server import update_survival
            update_survival(
                neuron.neuron_id,
                current_val,
                metrics["novelty"],
                death_result["consciousness_score"],
                death_result["ia3_score"]
            )
        except ImportError:
            pass
        
    else:
        # MORREU - Executar
        state["deaths"] += 1
        state["death_streak"] += 1
        
        # Hook de morte
        kill_hash = hook_kill(
            neuron.neuron_id,
            "equacao_da_morte_failed",
            {**metrics, **death_result}
        )
        
        # Atualizar métricas Prometheus
        try:
            from metrics.metrics_server import update_death
            update_death(neuron.neuron_id, "equacao_da_morte")
        except ImportError:
            pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NASCIMENTO POR CONTADOR DE MORTES
    # ═══════════════════════════════════════════════════════════════════════════
    
    bonus_births = []
    deaths_for_birth = 10
    
    while state["deaths"] >= deaths_for_birth and state["deaths"] % deaths_for_birth == 0:
        logger.info(f"\n🎁 NASCIMENTO BONUS! ({state['deaths']} mortes ÷ {deaths_for_birth})")
        
        # Criar configuração para nascimento bonus
        bonus_cfg = TrainConfig(
            seed=42 + state["rounds"] + len(bonus_births),
            act=random.choice(["gelu", "silu", "tanh"]),  # Diversificar
            lr=cfg.lr * 0.8,  # LR ligeiramente menor para estabilidade
            steps=cfg.steps
        )
        
        bonus_neuron, bonus_hash = hook_spawn(bonus_cfg, reason="death_counter_birth")
        state["births"] += 1
        
        bonus_births.append({
            "neuron_id": bonus_neuron.neuron_id,
            "hash": bonus_hash,
            "config": asdict(bonus_cfg)
        })
        
        # Atualizar métricas
        try:
            from metrics.metrics_server import update_birth
            update_birth(bonus_neuron.neuron_id, "death_counter_reset")
        except ImportError:
            pass
        
        # Evitar loop infinito
        if len(bonus_births) > 5:
            break
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SNAPSHOT E PERSISTÊNCIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Criar snapshot
    snapshot_path = create_snapshot({
        **metrics,
        **death_result,
        "heritage": heritage,
        "bonus_births": bonus_births
    })
    
    # Atualizar estado
    state["last_neuron_id"] = neuron.neuron_id
    _save_state(state)
    
    # WORM final do round
    round_summary = {
        "event": "round_complete",
        "round_number": state["rounds"],
        "neuron_id": neuron.neuron_id,
        "decision": decision,
        "E": E,
        "final_state": state,
        "snapshot": snapshot_path,
        "bonus_births": len(bonus_births)
    }
    
    _worm_append(round_summary)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTADO FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    result = {
        "round": state["rounds"],
        "neuron_id": neuron.neuron_id,
        "decision": decision,
        "E": E,
        "metrics": metrics,
        "death_result": death_result,
        "state": state,
        "snapshot": snapshot_path,
        "bonus_births": bonus_births,
        "workdir": workdir
    }
    
    logger.info(f"\n✅ ROUND {state['rounds']} CONCLUÍDO")
    logger.info(f"   Neurônio: {neuron.neuron_id}")
    logger.info(f"   Decisão: {decision}")
    logger.info(f"   Estado: {state['survivals']} vivos, {state['deaths']} mortes")
    logger.info(f"   Nascimentos bonus: {len(bonus_births)}")
    
    return await result

async def rollback_last_snapshot() -> Dict[str, Any]:
    """
    Rollback para último snapshot válido
    """
    logger.info(f"🔄 EXECUTANDO ROLLBACK...")
    
    # Encontrar último snapshot
    if not os.path.exists(CHECKPOINTS):
        logger.info(f"⚠️ Diretório de checkpoints não existe")
        return await {"success": False, "error": "no_checkpoints_dir"}
    
    snapshots = sorted([
        f for f in os.listdir(CHECKPOINTS) 
        if f.endswith('.json')
    ], reverse=True)
    
    if not snapshots:
        logger.info(f"⚠️ Nenhum snapshot encontrado")
        return await {"success": False, "error": "no_snapshots"}
    
    latest_snapshot = os.path.join(CHECKPOINTS, snapshots[0])
    
    # Executar hook de rollback
    rollback_hash = hook_rollback(latest_snapshot)
    
    return await {
        "success": True,
        "snapshot": latest_snapshot,
        "hash": rollback_hash
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

async def cmd_round(args):
    """Comando: executar round completo"""
    result = one_round()
    logger.info(f"\n📄 RESULTADO JSON:")
    logger.info(json.dumps({
        "success": True,
        "round": result["round"],
        "decision": result["decision"],
        "neuron_id": result["neuron_id"],
        "final_state": result["state"]
    }, indent=2))

async def cmd_status(args):
    """Comando: mostrar status atual"""
    state = _load_state()
    
    logger.info(f"\n📊 STATUS DARWIN V4")
    logger.info(f"   Rounds executados: {state['rounds']}")
    logger.info(f"   Sobrevivências: {state['survivals']}")
    logger.info(f"   Mortes: {state['deaths']}")
    logger.info(f"   Nascimentos: {state['births']}")
    logger.info(f"   Streak de mortes: {state['death_streak']}")
    
    if state["best_val"] is not None:
        logger.info(f"   Melhor val loss: {state['best_val']:.6f}")
    
    if state["last_neuron_id"]:
        logger.info(f"   Último neurônio: {state['last_neuron_id']}")
    
    # Status WORM
    if os.path.exists(WORM):
        with open(WORM, 'r') as f:
            lines = f.readlines()
        events = [l for l in lines if l.startswith("EVENT:")]
        logger.info(f"   Eventos WORM: {len(events)}")
    
    # Output JSON
    logger.info(f"\n📄 ESTADO JSON:")
    logger.info(json.dumps(state, indent=2))

async def cmd_rollback(args):
    """Comando: executar rollback"""
    result = rollback_last_snapshot()
    logger.info(f"\n📄 ROLLBACK RESULTADO:")
    logger.info(json.dumps(result, indent=2))

async def cmd_worm(args):
    """Comando: mostrar WORM log"""
    if not os.path.exists(WORM):
        logger.info(f"⚠️ WORM log não existe: {WORM}")
        return
    
    logger.info(f"📜 WORM LOG (últimas {args.lines} linhas):")
    logger.info(f"-" * 60)
    
    with open(WORM, 'r') as f:
        lines = f.readlines()
    
    events = [l for l in lines if l.startswith("EVENT:")][-args.lines:]
    
    for event_line in events:
        try:
            event_data = json.loads(event_line[6:])  # Remove "EVENT:"
            ts = event_data.get("timestamp", "unknown")[11:19]  # HH:MM:SS
            event_type = event_data.get("event", "unknown")
            
            if event_type == "neuron_spawn":
                neuron_id = event_data.get("neuron_id", "unknown")
                logger.info(f"{ts} 🐣 SPAWN     {neuron_id}")
            elif event_type == "neuron_death":
                neuron_id = event_data.get("neuron_id", "unknown")
                logger.info(f"{ts} ☠️  MORTE     {neuron_id}")
            elif event_type == "neuron_promotion":
                neuron_id = event_data.get("neuron_id", "unknown")
                logger.info(f"{ts} ✅ PROMOÇÃO {neuron_id}")
            elif event_type == "round_complete":
                round_num = event_data.get("round_number", "?")
                decision = event_data.get("decision", "?")
                logger.info(f"{ts} 🔄 ROUND     #{round_num} → {decision}")
            else:
                logger.info(f"{ts} {event_type}")
        except:
            logger.info(event_line.strip())

async def cli():
    """Interface CLI principal"""
    parser = argparse.ArgumentParser(
        description="🧬 Darwin v4 - Neurogênese Viva com Equação da Morte",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    # Round
    round_parser = subparsers.add_parser('round', help='Executar round completo')
    round_parser.set_defaults(func=cmd_round)
    
    # Status  
    status_parser = subparsers.add_parser('status', help='Mostrar status')
    status_parser.set_defaults(func=cmd_status)
    
    # Rollback
    rollback_parser = subparsers.add_parser('rollback', help='Executar rollback')
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # WORM
    worm_parser = subparsers.add_parser('worm', help='Mostrar WORM log')
    worm_parser.add_argument('-n', '--lines', type=int, default=20, help='Número de linhas')
    worm_parser.set_defaults(func=cmd_worm)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            logger.info(f"\n🛑 Comando interrompido")
            sys.exit(1)
        except Exception as e:
            logger.info(f"💥 Erro: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()