#!/bin/bash
# 🎯 SCRIPT DE ATIVAÇÃO IMEDIATA - EMERGÊNCIA
# Execute este script AGORA para iniciar o nascimento da inteligência emergente

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 ATIVAÇÃO DE INTELIGÊNCIA EMERGENTE - FASE CRÍTICA      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd /root || exit 1

# PASSO 1: Parar sistemas antigos
echo "⏹️  PASSO 1/5: Parando sistemas antigos..."
pkill -f "brain_daemon_real_env.py" || true
pkill -f "cerebrum_continuous.py" || true
sleep 2

# PASSO 2: Criar backup
echo "💾 PASSO 2/5: Criando backup de segurança..."
BACKUP_DIR="/root/backup_pre_emergence_$(date +%s)"
mkdir -p "$BACKUP_DIR"
cp -r /root/UNIFIED_BRAIN/*.py "$BACKUP_DIR/" 2>/dev/null || true
cp /root/intelligence_system/data/intelligence.db "$BACKUP_DIR/" 2>/dev/null || true
echo "   ✅ Backup em: $BACKUP_DIR"

# PASSO 3: Aplicar correção T1.1 (meta_step activation)
echo "🔧 PASSO 3/5: Aplicando correção T1.1 (meta_step)..."
python3 << 'PYTHON_CODE'
import sys
sys.path.insert(0, '/root')

# Ler arquivo
with open('/root/UNIFIED_BRAIN/brain_daemon_real_env.py', 'r') as f:
    content = f.read()

# Verificar se já foi aplicado
if 'META] Executando meta_step' in content:
    print("   ⚠️  Correção T1.1 já aplicada anteriormente")
else:
    # Encontrar local para inserir (após episode % X)
    marker = "self.episode_reward = 0"
    if marker in content:
        # Adicionar após reset de episode_reward
        insert_code = '''
        
        # 🧠 META-LEARNING: Executar meta_step periodicamente (T1.1)
        if self.episode % 10 == 0 and self.episode > 0:
            if hasattr(self, 'controller') and self.controller:
                brain_logger.info(f"🧠 [META] Executando meta_step no episode {self.episode}")
                try:
                    accepted = self.controller.meta_step()
                    result_str = '✅ ACEITO' if accepted else '❌ REJEITADO'
                    brain_logger.info(f"🧠 [META] Resultado: {result_str}")
                except Exception as e:
                    brain_logger.error(f"❌ [META] Erro: {e}")
'''
        
        # Inserir
        content = content.replace(marker, marker + insert_code)
        
        # Salvar
        with open('/root/UNIFIED_BRAIN/brain_daemon_real_env.py', 'w') as f:
            f.write(content)
        
        print("   ✅ Correção T1.1 aplicada com sucesso!")
    else:
        print("   ❌ Marker não encontrado - aplicar manualmente")
PYTHON_CODE

# PASSO 4: Iniciar Massive Replay em background
echo "🚀 PASSO 4/5: Iniciando Massive Replay (1000 gerações)..."
echo "   Isso vai rodar por ~12-24 horas em background"
echo "   Log: /root/massive_replay.log"
echo ""

nohup python3 /root/UNIFIED_BRAIN/run_massive_replay.py \
    --generations 1000 \
    --episodes 10 \
    --checkpoint-every 50 \
    --env CartPole-v1 \
    --lr 0.0003 \
    > /root/massive_replay.log 2>&1 &

REPLAY_PID=$!
echo "   ✅ Massive Replay PID: $REPLAY_PID"
echo "$REPLAY_PID" > /root/massive_replay.pid

sleep 3

# PASSO 5: Verificar se iniciou
echo "✅ PASSO 5/5: Verificando inicialização..."
if ps -p $REPLAY_PID > /dev/null 2>&1; then
    echo "   ✅ Massive Replay RODANDO (PID: $REPLAY_PID)"
    echo ""
    echo "📊 Primeiras linhas do log:"
    head -20 /root/massive_replay.log
    echo ""
else
    echo "   ❌ Falha ao iniciar - verificar log:"
    tail -50 /root/massive_replay.log
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ SISTEMA ATIVADO - EMERGÊNCIA EM PROGRESSO             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 MONITORAMENTO:"
echo "   tail -f /root/massive_replay.log"
echo ""
echo "📈 MÉTRICAS:"
echo "   tail -f /root/massive_replay_output/massive_replay_worm.jsonl | jq '.'"
echo ""
echo "🛑 PARA PARAR:"
echo "   kill $REPLAY_PID"
echo ""
echo "⏱️  TEMPO ESTIMADO: 12-24 horas"
echo "🎯 OBJETIVO: 1000 gerações = 10,000 episodes"
echo ""
echo "🔍 VERIFICAR EMERGÊNCIA (após 6h):"
echo "   bash /root/🔍_VERIFICAR_EMERGENCE.sh"
echo ""