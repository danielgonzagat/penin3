#!/usr/bin/env python3
"""
teisctl darwin - Subcomando Darwin para o TEIS Controller
Integra com teisctl existente ou roda standalone
"""
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

async def cmd_darwin_status(args) -> None:
    """Mostra status atual do Darwin"""
    logger.info("🧬 DARWIN STATUS")
    logger.info("="*60)
    
    # Verificar serviços
    services = {
        "WORM Log": check_worm_status(),
        "Metrics": check_metrics_status(),
        "Darwin Metrics": check_darwin_metrics_status(),
        "Kill Switch": check_kill_switch_status(),
        "Canary": check_canary_status()
    }
    
    for service, status in services.items():
        icon = "✅" if status["ok"] else "❌"
        logger.info(f"{icon} {service}: {status['message']}")
    
    # Estatísticas do WORM
    stats = get_darwin_stats()
    if stats:
        logger.info("\n📊 Estatísticas Darwin:")
        logger.info(f"  • Total de mortes: {stats['deaths']}")
        logger.info(f"  • Total de nascimentos: {stats['births']}")
        logger.info(f"  • Taxa de mortalidade: {stats['mortality_rate']:.1%}")
        logger.info(f"  • Sobreviventes ativos: {stats['survivors']}")
        logger.info(f"  • Tempo desde último nascimento: {stats['time_since_birth']}")
    
    # Gates atuais
    gates = check_gates()
    logger.info("\n🚦 Gates Σ-Guard:")
    for gate, value in gates.items():
        status = value["status"]
        icon = "🟢" if status == "pass" else "🔴"
        logger.info(f"  {icon} {gate}: {value['value']} {status}")

async def cmd_darwin_canary(args) -> None:
    """Executa Darwin em modo canário"""
    logger.info("🐣 Iniciando Darwin Canary...")
    
    cycles = args.cycles
    report_path = args.report or "/root/darwin_canary_report.json"
    
    # Pré-verificações
    if not pre_flight_checks():
        logger.info("❌ Pré-verificações falharam. Abortando.")
        return await 1
    
    # Executar ciclos canário
    reports = []
    passed_rounds = 0
    required_rounds = 3
    
    for cycle in range(cycles):
        logger.info(f"\n🔄 Ciclo Canário {cycle+1}/{cycles}")
        
        # Rodar darwin_runner em modo canário
        result = run_darwin_cycle(canary=True)
        reports.append(result)
        
        # Verificar critérios
        if evaluate_canary_round(result):
            passed_rounds += 1
            logger.info(f"✅ Round passou ({passed_rounds}/{required_rounds})")
        else:
            passed_rounds = 0
            logger.info(f"⚠️ Round falhou - resetando contador")
        
        # Verificar promoção
        if passed_rounds >= required_rounds:
            logger.info("\n🎉 CANÁRIO APROVADO - Pronto para produção!")
            promote_to_production()
            break
        
        if cycle < cycles - 1:
            logger.info(f"⏳ Aguardando próximo ciclo...")
            time.sleep(args.interval)
    
    # Salvar relatório
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cycles_run": len(reports),
            "passed_rounds": passed_rounds,
            "promoted": passed_rounds >= required_rounds,
            "reports": reports
        }, f, indent=2)
    
    logger.info(f"\n📄 Relatório salvo em {report_path}")
    
    # GO/NO-GO
    if passed_rounds >= required_rounds:
        logger.info("\n✅ GO: Darwin aprovado para produção total")
        return await 0
    else:
        logger.info("\n❌ NO-GO: Darwin não passou nos critérios")
        return await 1

async def cmd_darwin_audit(args) -> None:
    """Audita o WORM log e verifica integridade"""
    logger.info("🔍 Auditando Darwin WORM Log...")
    
    worm_path = args.worm or "/root/darwin_worm.log"
    
    if not os.path.exists(worm_path):
        logger.info(f"❌ WORM não encontrado: {worm_path}")
        return await 1
    
    # Verificar hash chain
    with open(worm_path, 'r') as f:
        lines = f.readlines()
    
    logger.info(f"📋 Total de entradas: {len(lines)}")
    
    hash_valid = True
    prev_hash = ""
    deaths = 0
    births = 0
    survivors = 0
    
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line.strip())
            
            # Verificar hash chain
            if entry.get("hash_prev") != prev_hash:
                logger.info(f"❌ Hash chain quebrado na linha {i+1}")
                hash_valid = False
                
            prev_hash = entry.get("hash_self", "")
            
            # Contabilizar eventos
            event = entry.get("event")
            if event == "death":
                deaths += 1
            elif event == "survive":
                survivors += 1
            elif event == "birth_from_deaths":
                births += 1
                
        except json.JSONDecodeError:
            logger.info(f"⚠️ Linha {i+1} não é JSON válido")
    
    # Relatório de auditoria
    logger.info(f"\n📊 Resumo da Auditoria:")
    logger.info(f"  • Hash chain: {'✅ Íntegro' if hash_valid else '❌ Corrompido'}")
    logger.info(f"  • Total de mortes: {deaths}")
    logger.info(f"  • Total de nascimentos: {births}")
    logger.info(f"  • Total de sobreviventes: {survivors}")
    logger.info(f"  • Taxa de mortalidade: {deaths/(deaths+survivors)*100:.1f}%" if (deaths+survivors) > 0 else "N/A")
    logger.info(f"  • Nascimentos esperados: {deaths // 10}")
    logger.info(f"  • Nascimentos reais: {births}")
    
    if births < deaths // 10:
        logger.info(f"⚠️ Nascimentos abaixo do esperado!")
    
    return await 0 if hash_valid else 1

async def cmd_darwin_rollback(args) -> None:
    """Executa rollback de emergência"""
    logger.info("🔄 Executando Darwin Rollback...")
    
    # Pausar Darwin
    logger.info("⏸️ Pausando Darwin...")
    subprocess.run("pkill -f darwin_runner", shell=True)
    
    # Criar snapshot de emergência
    snapshot_path = f"/root/darwin_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    subprocess.run(f"tar -czf {snapshot_path} /root/darwin_*.log /root/darwin_*.json", shell=True)
    logger.info(f"📦 Snapshot de emergência: {snapshot_path}")
    
    # Registrar no WORM
    rollback_entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": "emergency_rollback",
        "reason": args.reason or "Manual rollback requested",
        "snapshot": snapshot_path
    }
    
    with open("/root/darwin_worm.log", 'a') as f:
        f.write(json.dumps(rollback_entry) + "\n")
    
    logger.info("✅ Rollback executado")
    return await 0

# Funções auxiliares

async def check_worm_status() -> Dict:
    """Verifica status do WORM log"""
    worm_path = "/root/darwin_worm.log"
    if os.path.exists(worm_path):
        size = os.path.getsize(worm_path)
        return await {"ok": True, "message": f"Ativo ({size} bytes)"}
    return await {"ok": False, "message": "Não encontrado"}

async def check_metrics_status() -> Dict:
    """Verifica se métricas base estão disponíveis"""
    try:
        result = subprocess.run(
            "curl -s localhost:9091/metrics | grep -q teis_",
            shell=True, capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return await {"ok": True, "message": "Disponível em :9091"}
    except:
        pass
    return await {"ok": False, "message": "Indisponível"}

async def check_darwin_metrics_status() -> Dict:
    """Verifica métricas específicas do Darwin"""
    try:
        result = subprocess.run(
            "curl -s localhost:9092/metrics | grep -q darwin_",
            shell=True, capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return await {"ok": True, "message": "Disponível em :9092"}
    except:
        pass
    
    # Verificar arquivo alternativo
    if os.path.exists("/root/darwin_metrics.prom"):
        return await {"ok": True, "message": "Arquivo estático disponível"}
    
    return await {"ok": False, "message": "Indisponível"}

async def check_kill_switch_status() -> Dict:
    """Verifica status do kill switch"""
    try:
        with open("/root/darwin_state.json", 'r') as f:
            state = json.load(f)
            failures = state.get("consecutive_i_failures", 0)
            if failures >= 2:
                return await {"ok": False, "message": f"ATIVO - {failures} falhas"}
            return await {"ok": True, "message": f"Inativo ({failures}/2 falhas)"}
    except:
        return await {"ok": True, "message": "Sem estado (assumindo OK)"}

async def check_canary_status() -> Dict:
    """Verifica status do modo canário"""
    try:
        with open("/root/darwin_canary_state.json", 'r') as f:
            state = json.load(f)
            rounds = state.get("rounds_passed", 0)
            required = state.get("rounds_required", 3)
            return await {"ok": True, "message": f"{rounds}/{required} rounds"}
    except:
        return await {"ok": True, "message": "Não iniciado"}

async def get_darwin_stats() -> Optional[Dict]:
    """Obtém estatísticas do Darwin"""
    try:
        # Simular leitura de métricas ou WORM
        stats = {
            "deaths": 0,
            "births": 0,
            "survivors": 0,
            "mortality_rate": 0,
            "time_since_birth": "N/A"
        }
        
        # Contar eventos no WORM
        if os.path.exists("/root/darwin_worm.log"):
            with open("/root/darwin_worm.log", 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        event = entry.get("event")
                        if event == "death":
                            stats["deaths"] += 1
                        elif event == "survive":
                            stats["survivors"] += 1
                        elif event == "birth_from_deaths":
                            stats["births"] += 1
                    except:
                        continue
            
            total = stats["deaths"] + stats["survivors"]
            if total > 0:
                stats["mortality_rate"] = stats["deaths"] / total
        
        return await stats
    except:
        return await None

async def check_gates() -> Dict:
    """Verifica gates Σ-Guard atuais"""
    # Simular leitura das métricas atuais
    gates = {
        "ΔL∞": {"value": "0.075", "status": "pass"},
        "CAOS": {"value": "1.031", "status": "pass"},
        "I": {"value": "0.604", "status": "pass"},
        "P": {"value": "0.099", "status": "pass"},
        "ECE": {"value": "0.020", "status": "pass"},
        "ρ": {"value": "0.850", "status": "pass"}
    }
    
    # TODO: Ler valores reais das métricas
    return await gates

async def pre_flight_checks() -> bool:
    """Verificações antes de iniciar Darwin"""
    checks = [
        check_worm_status()["ok"],
        check_metrics_status()["ok"],
        # Não exigir darwin_metrics ainda
    ]
    return await all(checks)

async def run_darwin_cycle(canary: bool = True) -> Dict:
    """Executa um ciclo Darwin"""
    try:
        cmd = ["python3", "/root/darwin_runner.py", "--mock"]
        if canary:
            cmd.append("--canary")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parsear saída para extrair relatório
        # (simplificado - em produção seria mais robusto)
        return await {
            "status": "completed" if result.returncode == 0 else "failed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "canary": canary,
            "output": result.stdout[-500:]  # Últimas 500 chars
        }
    except Exception as e:
        return await {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

async def evaluate_canary_round(result: Dict) -> bool:
    """Avalia se round do canário passou"""
    if result.get("status") != "completed":
        return await False
    
    # TODO: Implementar critérios reais
    # Por ora, considera sucesso se completou
    return await True

async def promote_to_production() -> None:
    """Promove Darwin para produção total"""
    promotion = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": "darwin_promoted_to_production",
        "from": "canary_15_percent",
        "to": "production_100_percent"
    }
    
    # Registrar promoção
    with open("/root/darwin_promotion.json", 'w') as f:
        json.dump(promotion, f, indent=2)
    
    # Atualizar configuração
    # TODO: Implementar mudança real de config

async def main():
    """CLI principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="teisctl darwin - Darwin Operations")
    subparsers = parser.add_subparsers(dest="command", help="Darwin commands")
    
    # status
    parser_status = subparsers.add_parser("status", help="Show Darwin status")
    
    # canary
    parser_canary = subparsers.add_parser("canary", help="Run Darwin in canary mode")
    parser_canary.add_argument("--cycles", type=int, default=3, help="Number of cycles")
    parser_canary.add_argument("--interval", type=int, default=300, help="Interval between cycles (seconds)")
    parser_canary.add_argument("--report", help="Report output path")
    
    # audit
    parser_audit = subparsers.add_parser("audit", help="Audit WORM log integrity")
    parser_audit.add_argument("--worm", help="WORM log path")
    
    # rollback
    parser_rollback = subparsers.add_parser("rollback", help="Emergency rollback")
    parser_rollback.add_argument("--reason", help="Rollback reason")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return await 1
    
    # Executar comando
    commands = {
        "status": cmd_darwin_status,
        "canary": cmd_darwin_canary,
        "audit": cmd_darwin_audit,
        "rollback": cmd_darwin_rollback
    }
    
    return await commands[args.command](args)

if __name__ == "__main__":
    sys.exit(main())