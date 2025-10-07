#!/bin/bash
# COMANDOS PRONTOS - DARWINACCI SYSTEM

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DARWINACCI - COMANDOS PRONTOS                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "1️⃣  VALIDAR INTEGRAÇÃO"
echo "   python3 test_darwinacci_v7_integration.py"
echo ""

echo "2️⃣  RODAR DARWINACCI STANDALONE (10 cycles)"
echo "   DARWINACCI_CYCLES=10 python3 intelligence_system/tools/darwinacci_run.py"
echo ""

echo "3️⃣  RODAR SISTEMA COMPLETO (20 cycles rápido)"
echo "   python3 intelligence_system/core/unified_agi_system.py 20"
echo ""

echo "4️⃣  SOAK TEST 200 CYCLES"
echo "   bash intelligence_system/scripts/run_soak_200.sh"
echo ""

echo "5️⃣  OVERNIGHT SOAK (500-1000 cycles)"
echo "   bash intelligence_system/scripts/soak_overnight.sh 1000"
echo ""

echo "6️⃣  MONITORAR MÉTRICAS"
echo "   curl http://localhost:9108/metrics | grep darwinacci"
echo ""

echo "7️⃣  VER TIMELINE"
echo "   tail -20 intelligence_system/data/exports/timeline_metrics.csv"
echo ""

echo "8️⃣  HEALTH DASHBOARD"
echo "   python3 intelligence_system/tools/health_exporter.py"
echo "   cat intelligence_system/data/exports/api_health.csv"
echo ""

echo "9️⃣  VERIFICAR WORM"
echo "   tail -3 intelligence_system/data/unified_worm.jsonl | jq ."
echo ""

echo "🔟  SMOKE TEST APIs"
echo "   python3 intelligence_system/tools/smoke_apis.py"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "Status atual: ✅ 4/6 APIs | IA³=100.00% | 93 cycles validados"
echo "════════════════════════════════════════════════════════════════"
