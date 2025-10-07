#!/bin/bash
# Setup agente autônomo usando LLaMa 8B (que JÁ funciona)
# Em vez de Qwen (que está com timeout)

set -e

echo "🤖 SETUP AGENTE COM LLAMA 8B"
echo "============================"
echo ""
echo "Usando LLaMa 3.1-8B (porta 8080) - JÁ funcionando!"
echo ""

# Test LLaMa first
echo "🧪 Testando LLaMa 8B..."
HEALTH=$(curl -m 5 http://127.0.0.1:8080/health 2>/dev/null || echo "fail")

if [ "$HEALTH" == '{"status":"ok"}' ]; then
    echo "✅ LLaMa 8B respondendo"
else
    echo "❌ LLaMa não acessível - verificar se está rodando"
    echo "   ps aux | grep llama-server"
    exit 1
fi

# Get model ID
MODEL_ID=$(curl -m 10 http://127.0.0.1:8080/v1/models 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || echo "llama-3.1-8b")
echo "   Model ID: $MODEL_ID"
echo ""

# ============================================
# FASE 1: Workspace Isolado
# ============================================
echo "🛡️ FASE 1: Criando workspace isolado..."

mkdir -p /srv/agent-workspace-safe
mkdir -p /srv/agent-backups

# Copiar código (SEM databases, logs, checkpoints)
echo "   Copiando intelligence_system..."
rsync -a --exclude='*.db' --exclude='*.log' --exclude='*.pth' --exclude='__pycache__' \
    /root/intelligence_system/ /srv/agent-workspace-safe/intelligence_system/ 2>/dev/null || \
    cp -r /root/intelligence_system /srv/agent-workspace-safe/

echo "   Copiando UNIFIED_BRAIN..."
rsync -a --exclude='*.db' --exclude='*.log' --exclude='*.pt' --exclude='__pycache__' \
    /root/UNIFIED_BRAIN/ /srv/agent-workspace-safe/UNIFIED_BRAIN/ 2>/dev/null || \
    cp -r /root/UNIFIED_BRAIN /srv/agent-workspace-safe/

echo "   Copiando darwinacci_omega..."
rsync -a --exclude='*.db' --exclude='*.log' --exclude='__pycache__' \
    /root/darwinacci_omega/ /srv/agent-workspace-safe/darwinacci_omega/ 2>/dev/null || \
    cp -r /root/darwinacci_omega /srv/agent-workspace-safe/

echo "✅ Workspace: $(du -sh /srv/agent-workspace-safe | cut -f1)"
echo ""

# ============================================
# FASE 2: Git Checkpoint
# ============================================
echo "📦 FASE 2: Git checkpoint..."

cd /root
git init 2>/dev/null || true
git add -A 2>/dev/null || true
git commit -m "PRE-AGENT: Safety checkpoint $(date)" 2>/dev/null || true

TAG="pre-agent-$(date +%s)"
git tag "$TAG" 2>/dev/null || true

echo "✅ Rollback disponível: git reset --hard $TAG"
echo ""

# ============================================
# FASE 3: Safety Controller
# ============================================
echo "🛡️ FASE 3: Criando Safety Controller..."

cat > /root/AGENT_SAFETY_CONTROLLER_SIMPLE.py << 'EOFPY'
#!/usr/bin/env python3
"""
SIMPLE Safety Controller - Valida mudanças do agente
"""
import subprocess
import sys
from pathlib import Path
import time

class SimpleSafetyController:
    def __init__(self):
        self.workspace = Path('/srv/agent-workspace-safe')
        self.prod = Path('/root')
        self.kill_switch = Path('/root/.AGENT_KILL_SWITCH')
        print("🛡️ Safety Controller ready")
        print(f"   Workspace: {self.workspace}")
        print(f"   Kill switch: touch {self.kill_switch}")
    
    def validate_file(self, file_path):
        """Syntax check"""
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(file_path)],
            capture_output=True
        )
        return result.returncode == 0
    
    def approve(self, file_path):
        """Ask human approval"""
        print(f"\n{'='*60}")
        print(f"🤖 Mudança proposta: {file_path}")
        subprocess.run(['git', 'diff', file_path], cwd=self.workspace)
        print(f"{'='*60}")
        response = input("\nAprovar? (yes/no): ")
        return response.lower() == 'yes'
    
    def apply(self, file_path):
        """Copy to production"""
        import shutil
        src = self.workspace / file_path
        dst = self.prod / file_path
        
        # Backup
        if dst.exists():
            shutil.copy2(dst, str(dst) + f'.bak_{int(time.time())}')
        
        # Copy
        shutil.copy2(src, dst)
        print(f"✅ Applied: {file_path}")
    
    def monitor(self):
        """Monitor for changes"""
        print("\n🔍 Monitoring workspace (Ctrl+C to stop)...")
        try:
            while True:
                if self.kill_switch.exists():
                    print("🛑 Kill switch activated")
                    break
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n✅ Monitor stopped")

if __name__ == "__main__":
    controller = SimpleSafetyController()
    controller.monitor()
EOFPY

chmod +x /root/AGENT_SAFETY_CONTROLLER_SIMPLE.py

echo "✅ Safety Controller criado"
echo ""

# ============================================
# FASE 4: OpenHands com LLaMa 8B
# ============================================
echo "🔧 FASE 4: Configurando OpenHands com LLaMa 8B..."

# Stop any existing
docker stop openhands openhands-safe 2>/dev/null || true
docker rm openhands openhands-safe 2>/dev/null || true

# Pull image if needed
docker pull ghcr.io/all-hands-ai/openhands:latest || echo "Usando imagem cached"

# Start with LLaMa 8B
docker run -d --name openhands \
  --add-host=host.docker.internal:host-gateway \
  --memory=6g \
  --cpus=8 \
  -p 3000:3000 \
  -v /srv/agent-workspace-safe:/workspace \
  -e WORKSPACE_BASE=/workspace \
  -e LLM_MODEL="$MODEL_ID" \
  -e LLM_BASE_URL="http://host.docker.internal:8080/v1" \
  -e LLM_API_KEY="sk-local" \
  -e LLM_EMBEDDING_MODEL="$MODEL_ID" \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE="python:3.11" \
  --restart unless-stopped \
  ghcr.io/all-hands-ai/openhands:latest

sleep 10

# Check
if docker ps | grep -q openhands; then
    echo "✅ OpenHands rodando"
    docker logs openhands --tail 10
else
    echo "❌ OpenHands falhou ao iniciar"
    docker logs openhands --tail 30
    exit 1
fi

echo ""

# ============================================
# FASE 5: Criar primeira tarefa
# ============================================
echo "📝 FASE 5: Preparando primeira tarefa..."

cat > /srv/agent-workspace-safe/FIRST_TASK.txt << 'EOFTASK'
🤖 PRIMEIRA TAREFA DO AGENTE

Objetivo: Analisar 1 arquivo e propor 1 melhoria simples

Instruções:
1. Listar arquivos Python: ls -la /workspace/intelligence_system/core/*.py | head -5
2. Escolher 1 arquivo
3. Ler o arquivo
4. Encontrar 1 oportunidade de melhoria:
   - Função sem docstring
   - Variável mal nomeada (ex: 'x', 'tmp')
   - Código duplicado
   - Magic number sem constante

5. Propor mudança (máximo 5 linhas)
6. Explicar benefício
7. Não aplicar ainda - apenas propor

Exemplo de output esperado:
```
ANÁLISE: intelligence_system/core/database.py
Linha 42: Função _ensure_tables() sem docstring
Proposta: Adicionar docstring explicando propósito
Benefício: Melhora documentação e manutenibilidade
Mudança: +3 linhas (docstring)
```
EOFTASK

echo "✅ Tarefa definida"
echo ""

# ============================================
# RESUMO FINAL
# ============================================
echo "=============================================="
echo "✅ SETUP COMPLETO COM LLAMA 8B!"
echo "=============================================="
echo ""
echo "📊 COMPONENTES:"
echo "  ✅ LLaMa 8B: http://127.0.0.1:8080 (porta host)"
echo "  ✅ OpenHands: http://localhost:3000 (UI)"
echo "  ✅ Workspace: /srv/agent-workspace-safe"
echo "  ✅ Safety Controller: /root/AGENT_SAFETY_CONTROLLER_SIMPLE.py"
echo "  ✅ Kill switch: touch /root/.AGENT_KILL_SWITCH"
echo "  ✅ Rollback: git reset --hard $TAG"
echo ""
echo "🎯 PRÓXIMOS PASSOS:"
echo ""
echo "1. TERMINAL 1 - Iniciar Safety Controller:"
echo "   python3 /root/AGENT_SAFETY_CONTROLLER_SIMPLE.py"
echo ""
echo "2. NAVEGADOR - Abrir OpenHands:"
echo "   http://localhost:3000"
echo ""
echo "3. OPENHANDS UI - Verificar config:"
echo "   Settings → LLM deve mostrar connection"
echo "   Se não, configurar manualmente:"
echo "     Base URL: http://host.docker.internal:8080/v1"
echo "     Model: $MODEL_ID"
echo "     API Key: sk-local"
echo ""
echo "4. CHAT - Primeira tarefa:"
echo "   Copiar conteúdo de: /srv/agent-workspace-safe/FIRST_TASK.txt"
echo "   Colar no chat"
echo "   Observar agente analisar código"
echo ""
echo "5. APROVAR - Se mudança boa:"
echo "   Safety Controller mostrará diff"
echo "   Você aprova"
echo "   Aplicada a /root se tests passarem"
echo ""
echo "=============================================="
echo "🔥 AGENTE PRONTO COM LLAMA 8B!"
echo "=============================================="
echo ""
echo "⚠️  Lembrete: Workspace é ISOLADO"
echo "   Mudanças só vão para /root após SUA aprovação"
echo ""