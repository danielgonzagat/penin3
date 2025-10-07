#!/bin/bash
# TEIS MAXIMUM INTELLIGENCE RUNNER
# Script autônomo para executar TEIS com máxima chance de inteligência emergir

echo "🧬 TEIS MAXIMUM INTELLIGENCE SYSTEM"
echo "==================================="
echo "Iniciando em $(date)"
echo ""

# Configurações
LOGDIR="/root/teis_logs"
CHECKPOINT_DIR="/root/teis_checkpoints"
CYCLES=10000
MONITOR_INTERVAL=100

# Criar diretórios necessários
mkdir -p "$LOGDIR"
mkdir -p "$CHECKPOINT_DIR"

# Timestamp para esta execução
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/teis_run_$TIMESTAMP.log"
EMERGENCE_LOG="$LOGDIR/emergence_$TIMESTAMP.jsonl"

echo "📁 Logs em: $LOGFILE"
echo "📊 Emergências em: $EMERGENCE_LOG"
echo ""

# Python script inline para máxima integração
python3 << 'PYTHON_END' 2>&1 | tee "$LOGFILE"
import sys
import os
import time
import json
import signal
from datetime import datetime

# Adicionar root ao path
sys.path.insert(0, '/root')

print("🔧 Importando módulos...")

try:
    from teis_enhanced import TEISEnhanced
    print("✅ TEIS Enhanced carregado")
except Exception as e:
    print(f"❌ Erro ao carregar TEIS Enhanced: {e}")
    sys.exit(1)

# Tentar carregar módulos opcionais
optional_modules = []

try:
    from swarm_intelligence import SwarmIntelligence
    optional_modules.append("SwarmIntelligence")
except:
    print("⚠️ SwarmIntelligence não disponível")

try:
    from symbol_grounding import SymbolGroundingSystem
    optional_modules.append("SymbolGrounding")
except:
    print("⚠️ SymbolGrounding não disponível")

try:
    from quantum_processing import QuantumInspiredProcessor
    optional_modules.append("QuantumProcessing")
except:
    print("⚠️ QuantumProcessing não disponível")

print(f"\n🔌 Módulos opcionais carregados: {optional_modules}")

# Configuração do sistema (com overrides por env)
cycles_env = int(os.environ.get('TEIS_CYCLES', '10000'))
monitor_env = int(os.environ.get('TEIS_MONITOR', '100'))

CONFIG = {
    'num_agents': int(os.environ.get('TEIS_NUM_AGENTS', '30')),
    'cycles': cycles_env,
    'checkpoint_interval': int(os.environ.get('TEIS_CHECKPOINT_INTERVAL', '500')),
    'monitor_interval': monitor_env,
    'emergence_threshold': int(os.environ.get('TEIS_EMERGENCE_THRESHOLD', '5')),
    'symbol_threshold': int(os.environ.get('TEIS_SYMBOL_THRESHOLD', '50')),
    'task_pressure': os.environ.get('TEIS_TASK_PRESSURE', '1') not in ('0','false','False'),
    'resource_scarcity': float(os.environ.get('TEIS_RESOURCE_SCARCITY', '0.3')),
    'evolution_pressure': float(os.environ.get('TEIS_EVOLUTION_PRESSURE', '0.7')),
    'ctde': os.environ.get('TEIS_CTDE', '1'),
    'wm_rollouts': os.environ.get('TEIS_WM_ROLLOUTS', '1'),
}

print("\n🎮 Configuração do Sistema:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# Handler para salvar estado ao interromper
def signal_handler(signum, frame):
    print("\n\n🛑 Interrupção detectada. Salvando estado...")
    try:
        system.save_checkpoint(f'/root/teis_checkpoints/emergency_save_{timestamp}.json')
        print("✅ Estado salvo com sucesso")
    except:
        print("❌ Erro ao salvar estado")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Criar sistema
print("\n🌟 Criando sistema TEIS Enhanced...")
system = TEISEnhanced(num_agents=CONFIG['num_agents'])

# Aplicar pressão evolutiva se configurado
if CONFIG['resource_scarcity'] < 1.0:
    for resource in system.environment.resources:
        system.environment.resources[resource] *= CONFIG['resource_scarcity']
    print(f"⚡ Recursos reduzidos para {CONFIG['resource_scarcity']*100:.0f}%")

# Variáveis de monitoramento
metrics = {
    'start_time': time.time(),
    'total_emergences': 0,
    'unique_emergence_types': set(),
    'max_symbols': 0,
    'tasks_completed': 0,
    'agent_deaths': 0,
    'intelligence_moments': [],
    'cultural_events': [],
    'highest_q_value': 0,
    'lowest_exploration_rate': 1.0
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
emergence_file = f'/root/teis_logs/emergence_{timestamp}.jsonl'

print("\n🚀 Iniciando execução principal...")
print("=" * 60)

# Loop principal
for cycle in range(CONFIG['cycles']):
    try:
        # Executar ciclo
        cycle_data = system.run_cycle()
        
        # Atualizar métricas
        metrics['total_emergences'] += len(cycle_data['emergent_behaviors'])
        
        for behavior in cycle_data['emergent_behaviors']:
            metrics['unique_emergence_types'].add(behavior['type'])
            
            # Salvar emergências significativas
            if behavior['level'] in ['meso', 'macro']:
                with open(emergence_file, 'a') as f:
                    f.write(json.dumps(behavior) + '\n')
        
        metrics['max_symbols'] = max(metrics['max_symbols'], cycle_data['symbols_discovered'])
        metrics['tasks_completed'] += cycle_data['tasks_completed']
        
        # Verificar Q-values máximos (aprendizado)
        for agent in system.agents:
            if agent.q_table:
                max_q = max(max(actions.values()) for actions in agent.q_table.values())
                metrics['highest_q_value'] = max(metrics['highest_q_value'], max_q)
                metrics['lowest_exploration_rate'] = min(metrics['lowest_exploration_rate'], agent.exploration_rate)
        
        # Inserir tarefa referencial periodicamente (protocolo de grounding)
        if cycle % 75 == 0:
            try:
                from random import choice
                t = system.task_environment.generate_task(cycle)
                # força jogo referencial a cada 75 ciclos
                t['type'] = 'reference_game'
                t['description'] = 'Communicate a target so partner seeks correct resource'
                t['requires'] = ['communicate','seek_resource']
                t['min_agents'] = 2
                t['reward'] = 1.6
                if 'target_resource' not in t:
                    t['target_resource'] = choice(['food','water','shelter','materials'])
                system.environment.active_tasks.append(t)
            except Exception as _e:
                pass

        # Monitoramento periódico
        if cycle % CONFIG['monitor_interval'] == 0:
            elapsed = time.time() - metrics['start_time']
            emergences_per_second = metrics['total_emergences'] / elapsed
            
            print(f"\n📊 CICLO {cycle} | Tempo: {elapsed:.1f}s")
            print(f"   Emergências: {metrics['total_emergences']} total, {len(metrics['unique_emergence_types'])} tipos únicos")
            print(f"   Símbolos: {metrics['max_symbols']}")
            print(f"   Tarefas completadas: {metrics['tasks_completed']}")
            print(f"   Q-value máximo: {metrics['highest_q_value']:.3f}")
            print(f"   Taxa exploração mínima: {metrics['lowest_exploration_rate']:.3f}")
            print(f"   Taxa emergência: {emergences_per_second:.2f}/s")
            # CTDE/WM métricas
            try:
                print(f"   CTDE: entropy={getattr(system,'ctde_metrics',{}).get('avg_attention_entropy',0):.3f} avg_bonus={getattr(system,'ctde_metrics',{}).get('avg_ctde_bonus',0):.3f} team_value={getattr(system,'ctde_metrics',{}).get('team_value',0):.3f}")
            except Exception:
                pass
            
            # Verificar sinais de inteligência
            intelligence_score = 0
            
            if len(metrics['unique_emergence_types']) >= CONFIG['emergence_threshold']:
                intelligence_score += 1
                print("   ✓ Diversidade emergente alta")
            
            if metrics['max_symbols'] >= CONFIG['symbol_threshold']:
                intelligence_score += 1
                print("   ✓ Proto-linguagem detectada")
            
            if metrics['highest_q_value'] > 5.0:
                intelligence_score += 1
                print("   ✓ Aprendizado profundo detectado")
            
            if metrics['lowest_exploration_rate'] < 0.1:
                intelligence_score += 1
                print("   ✓ Exploração convergindo (conhecimento estável)")
            
            if 'self_organization' in metrics['unique_emergence_types']:
                intelligence_score += 1
                print("   ✓ AUTO-ORGANIZAÇÃO DETECTADA!")
            
            if 'cultural_emergence' in metrics['unique_emergence_types']:
                intelligence_score += 1
                print("   ✓ CULTURA EMERGENTE DETECTADA!")

            # Estatísticas do jogo referencial e MI proxy
            ref = cycle_data.get('reference_game',{})
            ref_trials = ref.get('trials_total',0)
            ref_success = ref.get('success_total',0)
            mi_proxy = 0.0
            try:
                # proxy: taxa de sucesso em jogos referenciais no período
                mi_proxy = (ref_success/(ref_trials+1e-6)) if ref_trials>0 else 0.0
            except Exception:
                mi_proxy = 0.0
            print(f"   ReferenceGame: trials={ref_trials} success={ref_success} mi_proxy={mi_proxy:.2f}")
                
            if intelligence_score >= 4:
                print("\n🧠 INTELIGÊNCIA EMERGENTE PROVÁVEL!")
                metrics['intelligence_moments'].append({
                    'cycle': cycle,
                    'score': intelligence_score,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Checkpoint periódico
        if cycle % CONFIG['checkpoint_interval'] == 0:
            checkpoint_file = f"/root/teis_checkpoints/checkpoint_{timestamp}_cycle_{cycle}.json"
            system.save_checkpoint(checkpoint_file)
            print(f"💾 Checkpoint salvo: {checkpoint_file}")
        
        # Adicionar pressão evolutiva crescente
        if cycle % 1000 == 0 and cycle > 0:
            # Aumentar dificuldade
            system.environment.social_tension = min(1.0, system.environment.social_tension + 0.1)
            # Reduzir recursos
            for resource in system.environment.resources:
                system.environment.resources[resource] *= 0.9
            print(f"⚡ Pressão evolutiva aumentada no ciclo {cycle}")
    
    except Exception as e:
        print(f"⚠️ Erro no ciclo {cycle}: {e}")
        continue

# Relatório final
print("\n" + "=" * 60)
print("📈 RELATÓRIO FINAL")
print("=" * 60)

elapsed_total = time.time() - metrics['start_time']
print(f"⏱️ Tempo total: {elapsed_total:.2f} segundos")
print(f"🔄 Ciclos executados: {CONFIG['cycles']}")
print(f"✨ Emergências totais: {metrics['total_emergences']}")
print(f"📊 Tipos únicos: {len(metrics['unique_emergence_types'])}")
print(f"🔤 Símbolos criados: {metrics['max_symbols']}")
print(f"✅ Tarefas completadas: {metrics['tasks_completed']}")
print(f"🧠 Momentos de inteligência detectados: {len(metrics['intelligence_moments'])}")

print("\n📋 Tipos de emergência observados:")
for emergence_type in sorted(metrics['unique_emergence_types']):
    print(f"   • {emergence_type}")

if metrics['intelligence_moments']:
    print("\n🎯 Momentos de inteligência:")
    for moment in metrics['intelligence_moments']:
        print(f"   Ciclo {moment['cycle']}: Score {moment['score']}/6")

# Salvar relatório final
report = {
    'timestamp': timestamp,
    'config': CONFIG,
    'metrics': {k: v if not isinstance(v, set) else list(v) for k, v in metrics.items()},
    'elapsed_time': elapsed_total,
    'emergence_types': list(metrics['unique_emergence_types']),
    'intelligence_detected': len(metrics['intelligence_moments']) > 0
}

report_file = f"/root/teis_logs/report_{timestamp}.json"
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Relatório completo salvo em: {report_file}")

# Salvar checkpoint final
final_checkpoint = f"/root/teis_checkpoints/final_{timestamp}.json"
system.save_checkpoint(final_checkpoint)
print(f"💾 Checkpoint final: {final_checkpoint}")

# Análise de inteligência
if len(metrics['unique_emergence_types']) >= 10 and metrics['max_symbols'] >= 100:
    print("\n🧠 VEREDITO: SINAIS FORTES DE INTELIGÊNCIA EMERGENTE!")
elif len(metrics['unique_emergence_types']) >= 5 and metrics['max_symbols'] >= 50:
    print("\n📈 VEREDITO: INTELIGÊNCIA EM DESENVOLVIMENTO")
else:
    print("\n🔄 VEREDITO: SISTEMA PRECISA MAIS TEMPO OU PRESSÃO")

print("\n✅ Execução completa!")

PYTHON_END

echo ""
echo "🏁 Script finalizado em $(date)"
echo "📊 Verifique os logs em: $LOGDIR"