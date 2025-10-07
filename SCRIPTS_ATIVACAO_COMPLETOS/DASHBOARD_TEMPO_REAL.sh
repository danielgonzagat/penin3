#!/bin/bash
# DASHBOARD EM TEMPO REAL DO SISTEMA I³
# ======================================
# Mostra status de todos componentes em uma tela

clear

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║  🧠 SISTEMA I³ - DASHBOARD EM TEMPO REAL                      ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo
    date
    echo
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧬 DARWIN EVOLUTION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Darwin status
    if pgrep -f "run_emergence_blocks_STORM" > /dev/null; then
        DARWIN_COUNT=$(pgrep -f "run_emergence_blocks_STORM" | wc -l)
        echo "✅ Status: ATIVO ($DARWIN_COUNT processos)"
        
        # Última geração
        LAST_GEN=$(tail -100 /root/darwin_STORM.log 2>/dev/null | grep -oP "Geração \K\d+" | tail -1)
        if [ -n "$LAST_GEN" ]; then
            echo "📊 Geração atual: $LAST_GEN/100"
        fi
        
        # Checkpoints recentes
        RECENT_CP=$(find /root/intelligence_system/models/darwin_checkpoints/ -name "*.pt" -mmin -30 2>/dev/null | wc -l)
        echo "🧬 Checkpoints (30min): $RECENT_CP novos"
    else
        echo "❌ Status: INATIVO"
    fi
    echo
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 SURPRISE DETECTOR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if pgrep -f "EMERGENCE_CATALYST_1_SURPRISE" > /dev/null; then
        echo "✅ Status: ATIVO"
        
        # Surpresas detectadas
        SURPRISES=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises" 2>/dev/null || echo 0)
        echo "🎉 Total surpresas: $SURPRISES"
        
        SURPRISES_1H=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises WHERE timestamp > datetime('now', '-1 hour')" 2>/dev/null || echo 0)
        echo "⏱️  Última hora: $SURPRISES_1H"
        
        # Top surpresa
        TOP_SURPRISE=$(sqlite3 /root/emergence_surprises.db "SELECT system || '.' || metric || ': ' || ROUND(surprise_score, 2) || 'σ' FROM surprises ORDER BY surprise_score DESC LIMIT 1" 2>/dev/null || echo "n/a")
        echo "🏆 Top: $TOP_SURPRISE"
    else
        echo "❌ Status: INATIVO"
    fi
    echo
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔗 OUTROS COMPONENTES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Meta-Learner
    if pgrep -f "META_LEARNER_REALTIME" > /dev/null; then
        echo "✅ Meta-Learner: ATIVO"
    else
        echo "❌ Meta-Learner: INATIVO"
    fi
    
    # Auto-Validator
    if pgrep -f "AUTO_VALIDATOR" > /dev/null; then
        echo "✅ Auto-Validator: ATIVO"
    else
        echo "❌ Auto-Validator: INATIVO"
    fi
    
    # System Connector
    if pgrep -f "SYSTEM_CONNECTOR" > /dev/null; then
        echo "✅ System Connector: ATIVO"
    else
        echo "❌ System Connector: INATIVO"
    fi
    
    # Consciousness
    if pgrep -f "CONSCIOUSNESS" > /dev/null; then
        AWARENESS=$(sqlite3 /root/consciousness.db "SELECT value FROM self_knowledge WHERE key='num_components' ORDER BY timestamp DESC LIMIT 1" 2>/dev/null || echo "n/a")
        echo "✅ Consciousness: ATIVO (${AWARENESS} componentes)"
    else
        echo "❌ Consciousness: INATIVO"
    fi
    
    # Cross-Pollination
    if pgrep -f "CROSS_POLLINATION" > /dev/null; then
        echo "✅ Cross-Pollination: ATIVO"
    else
        echo "❌ Cross-Pollination: INATIVO"
    fi
    echo
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 MÉTRICAS V7"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/intelligence_system/data/intelligence.db "SELECT 'Cycle ' || cycle || ': MNIST=' || ROUND(mnist_accuracy, 2) || '%, CartPole=' || ROUND(cartpole_reward, 1) FROM cycles ORDER BY cycle DESC LIMIT 3" 2>/dev/null || echo "n/a"
    echo
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Atualizado a cada 5s | Ctrl+C para sair"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sleep 5
done
