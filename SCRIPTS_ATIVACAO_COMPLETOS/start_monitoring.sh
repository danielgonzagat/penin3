#!/bin/bash
# Este script centraliza o monitoramento do UNIFIED_BRAIN, a inteligência ativa.
# Execute-o para ter uma visão completa e em tempo real do aprendizado.

echo "--- INICIANDO MONITORAMENTO CENTRALIZADO DO UNIFIED_BRAIN ---"

# Terminal 1: Log de Aprendizado (Apenas as vitórias e progressos)
gnome-terminal -- /bin/bash -c 'echo "--- LOG DE APRENDIZADO (RECOMPENSAS) ---"; tail -f /root/UNIFIED_BRAIN/logs/unified_brain.log | grep --line-buffered "NEW BEST" --color=always; read'

# Terminal 2: Dashboard de Métricas (Visão geral do estado do cérebro)
gnome-terminal -- /bin/bash -c 'echo "--- DASHBOARD DE MÉTRICAS ---"; watch -n 5 cat /root/UNIFIED_BRAIN/metrics_dashboard.json; read'

# Terminal 3: Log Completo (Para debugging e análise profunda)
gnome-terminal -- /bin/bash -c 'echo "--- LOG COMPLETO DO CÉREBRO ---"; tail -f /root/UNIFIED_BRAIN/logs/unified_brain.log; read'

echo "--- Monitores iniciados em novas janelas. ---"
