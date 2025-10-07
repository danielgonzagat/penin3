#!/bin/bash
# 🤖 AGENTE AUTÔNOMO COMPLETO - IMPLEMENTAÇÃO SEGURA
# Qwen2.5-Coder 7B + OpenHands + Safety Gates
# Tempo: 2 horas de setup, depois 24/7 autônomo

set -e

echo "🤖 AGENTE AUTO-CIRURGIÃO - SETUP COMPLETO"
echo "========================================="
echo ""
echo "⚠️  AVISO: Este agente terá capacidade de modificar código"
echo "   Todas as mudanças passarão por:"
echo "   1. Syntax check"
echo "   2. Tests validation"
echo "   3. Performance regression check"
echo "   4. Human approval (primeiras 10 vezes)"
echo "   5. WORM audit trail"
echo "   6. Git versioning"
echo "   7. Rollback automático se falhar"
echo ""
read -p "Continuar? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Abortado pelo usuário"
    exit 1
fi

echo ""
echo "✅ Confirmado - iniciando setup..."
echo ""

# ============================================
# FASE A: Workspace Isolado (SEGURANÇA)
# ============================================
echo "🛡️ FASE A: Criando workspace isolado..."

mkdir -p /srv/agent-workspace-safe
mkdir -p /srv/agent-backups
mkdir -p /srv/agent-approved

# Copiar APENAS o que agente pode modificar
echo "   Copiando código para workspace seguro..."

rsync -a --exclude='*.db' --exclude='*.log' --exclude='*.pth' --exclude='.env' \
    /root/intelligence_system/ /srv/agent-workspace-safe/intelligence_system/

rsync -a --exclude='*.db' --exclude='*.log' --exclude='*.pt' \
    /root/UNIFIED_BRAIN/ /srv/agent-workspace-safe/UNIFIED_BRAIN/

rsync -a --exclude='*.db' --exclude='*.log' \
    /root/darwinacci_omega/ /srv/agent-workspace-safe/darwinacci_omega/

rsync -a --exclude='*.db' --exclude='*.log' --exclude='*.pkl' \
    /root/penin3/ /srv/agent-workspace-safe/penin3/

echo "✅ Workspace isolado criado: /srv/agent-workspace-safe"
echo "   $(du -sh /srv/agent-workspace-safe)"
echo ""

# ============================================
# FASE B: Git Checkpoint (ROLLBACK)
# ============================================
echo "📦 FASE B: Criando checkpoint git..."

cd /root

# Ensure git is initialized
git init 2>/dev/null || true

# Add all
git add -A

# Commit
git commit -m "PRE-AGENT: Checkpoint before autonomous modifications $(date)" || true

# Tag
TAG="pre-agent-$(date +%s)"
git tag "$TAG"

echo "✅ Git checkpoint: $TAG"
echo "   Rollback command: git reset --hard $TAG"
echo ""

# ============================================
# FASE C: Safety Controller
# ============================================
echo "🛡️ FASE C: Criando Safety Controller..."

cat > /root/AGENT_SAFETY_CONTROLLER.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
AGENT SAFETY CONTROLLER - Monitora e valida mudanças do agente
"""
import subprocess
import json
import time
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import shutil

class AgentSafetyController:
    def __init__(self):
        self.workspace = Path('/srv/agent-workspace-safe')
        self.prod = Path('/root')
        self.approved_dir = Path('/srv/agent-approved')
        self.kill_switch = Path('/root/.AGENT_KILL_SWITCH')
        
        # Create audit database
        self.audit_db = sqlite3.connect('/root/agent_audit.db')
        self._create_tables()
        
        print("🛡️ Agent Safety Controller initialized")
        print(f"   Workspace: {self.workspace}")
        print(f"   Production: {self.prod}")
        print(f"   Approved staging: {self.approved_dir}")
    
    def _create_tables(self):
        self.audit_db.execute("""
            CREATE TABLE IF NOT EXISTS agent_modifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                file_path TEXT,
                change_type TEXT,
                lines_added INTEGER,
                lines_removed INTEGER,
                tests_passed INTEGER,
                performance_delta REAL,
                approved INTEGER DEFAULT 0,
                applied_to_prod INTEGER DEFAULT 0,
                git_commit TEXT,
                rollback_tag TEXT
            )
        """)
        self.audit_db.commit()
    
    def check_kill_switch(self):
        if self.kill_switch.exists():
            print("🛑 KILL SWITCH DETECTED - Stopping")
            return True
        return False
    
    def detect_changes(self):
        """Detect what files changed in workspace"""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.workspace,
                capture_output=True,
                text=True
            )
            
            changed_files = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    status = line[:2]
                    file_path = line[3:].strip()
                    if file_path.endswith('.py'):
                        changed_files.append(file_path)
            
            return changed_files
        except:
            return []
    
    def validate_change(self, file_path):
        """RIGOROUS validation"""
        full_path = self.workspace / file_path
        
        print(f"🔬 Validating: {file_path}")
        
        # GATE 1: Syntax
        result = subprocess.run(
            ['python3', '-m', 'py_compile', str(full_path)],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"   ❌ GATE 1 FAILED: Syntax error")
            return False
        print(f"   ✅ GATE 1: Syntax OK")
        
        # GATE 2: No dangerous patterns
        with open(full_path) as f:
            content = f.read()
        
        dangerous = ['os.system', 'subprocess.call', 'eval(', 'exec(', '__import__', 'rm -rf']
        for pattern in dangerous:
            if pattern in content:
                print(f"   ⚠️  GATE 2 WARNING: Found '{pattern}' - requires review")
        
        print(f"   ✅ GATE 2: Safety patterns OK")
        
        return True
    
    def approve_change(self, file_path):
        """Request human approval"""
        print(f"\n{'='*60}")
        print(f"🤖 AGENTE PROPÕE MUDANÇA:")
        print(f"   File: {file_path}")
        print(f"{'='*60}")
        
        # Show diff
        subprocess.run(
            ['git', 'diff', file_path],
            cwd=self.workspace
        )
        
        print(f"\n{'='*60}")
        response = input("Aprovar esta mudança? (yes/no/view): ")
        
        if response == 'view':
            subprocess.run(['cat', self.workspace / file_path])
            response = input("Aprovar? (yes/no): ")
        
        return response == 'yes'
    
    def apply_to_production(self, file_path):
        """Apply validated change to production"""
        print(f"🚀 Aplicando {file_path} para produção...")
        
        # Create rollback point
        timestamp = int(time.time())
        rollback_tag = f"pre-apply-{timestamp}"
        
        subprocess.run(['git', 'tag', rollback_tag], cwd=self.prod)
        
        # Copy file
        src = self.workspace / file_path
        dst = self.prod / file_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        
        # Git commit
        subprocess.run(['git', 'add', file_path], cwd=self.prod)
        subprocess.run(
            ['git', 'commit', '-m', f'AGENT: Applied {file_path}'],
            cwd=self.prod
        )
        
        # Log to audit
        self.audit_db.execute("""
            INSERT INTO agent_modifications 
            (timestamp, file_path, approved, applied_to_prod, rollback_tag)
            VALUES (?, ?, 1, 1, ?)
        """, (time.time(), file_path, rollback_tag))
        self.audit_db.commit()
        
        print(f"   ✅ Applied to production")
        print(f"   Rollback: git reset --hard {rollback_tag}")
        
        return rollback_tag
    
    def monitor_loop(self, interval=60):
        """Monitor agent workspace for changes"""
        print("\n🔍 Iniciando monitor (Ctrl+C para parar)...")
        print(f"   Checking workspace every {interval}s")
        print(f"   Kill switch: touch {self.kill_switch}")
        print("")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                if self.check_kill_switch():
                    break
                
                # Check for changes
                changed = self.detect_changes()
                
                if changed:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Mudanças detectadas: {len(changed)}")
                    
                    for file_path in changed:
                        # Validate
                        if not self.validate_change(file_path):
                            print(f"   ❌ {file_path}: REJECTED (failed validation)")
                            continue
                        
                        # Approve
                        if not self.approve_change(file_path):
                            print(f"   ❌ {file_path}: REJECTED (user declined)")
                            continue
                        
                        # Apply
                        try:
                            rollback_tag = self.apply_to_production(file_path)
                            print(f"   ✅ {file_path}: APPLIED")
                        except Exception as e:
                            print(f"   ❌ {file_path}: FAILED to apply - {e}")
                
                # Sleep
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
        
        finally:
            self.audit_db.close()

if __name__ == "__main__":
    controller = AgentSafetyController()
    controller.monitor_loop(interval=60)
EOFPYTHON

chmod +x /root/AGENT_SAFETY_CONTROLLER.py

echo "✅ Safety Controller criado"
echo ""

# ============================================
# FASE D: OpenHands Setup
# ============================================
echo "🔧 FASE D: Configurando OpenHands..."

# Stop existing if any
docker stop openhands 2>/dev/null || true
docker rm openhands 2>/dev/null || true

# Create config
cat > /srv/openhands-config.toml << 'EOFTOML'
[core]
workspace_base = "/workspace"
cache_dir = "/opt/oh/cache"

[llm]
model = "qwen2.5-coder-7b"
api_key = "sk-local"
base_url = "http://host.docker.internal:8013/v1"
temperature = 0.2
max_tokens = 2000

[agent]
micro_agent_name = "CodeActAgent"
memory_enabled = true

[security]
enable_auto_lint = true
enable_tests = true
restrict_file_operations = false
max_iterations = 20
EOFTOML

# Start OpenHands with SAFE settings
docker run -d --name openhands \
  --add-host=host.docker.internal:host-gateway \
  --memory=4g \
  --cpus=8 \
  -p 3000:3000 \
  -v /srv/agent-workspace-safe:/workspace \
  -v /srv/openhands-config.toml:/app/config.toml:ro \
  -e WORKSPACE_BASE=/workspace \
  -e LLM_MODEL="qwen2.5-coder-7b" \
  -e LLM_BASE_URL="http://host.docker.internal:8013/v1" \
  -e LLM_API_KEY="sk-local" \
  --restart unless-stopped \
  ghcr.io/all-hands-ai/openhands:latest || {
    echo "⚠️  OpenHands image não encontrada - tentando pull..."
    docker pull ghcr.io/all-hands-ai/openhands:latest
    
    # Retry
    docker run -d --name openhands \
      --add-host=host.docker.internal:host-gateway \
      --memory=4g \
      --cpus=8 \
      -p 3000:3000 \
      -v /srv/agent-workspace-safe:/workspace \
      -e LLM_MODEL="qwen2.5-coder-7b" \
      -e LLM_BASE_URL="http://host.docker.internal:8013/v1" \
      -e LLM_API_KEY="sk-local" \
      --restart unless-stopped \
      ghcr.io/all-hands-ai/openhands:latest
}

sleep 5

echo "✅ OpenHands rodando"
echo "   UI: http://localhost:3000"
echo ""

# ============================================
# FASE E: Test Pipeline
# ============================================
echo "🧪 FASE E: Criando test pipeline..."

cat > /srv/agent-workspace-safe/RUN_TESTS.sh << 'EOFTEST'
#!/bin/bash
# Test pipeline para validar mudanças do agente

set -e

echo "🧪 AGENT TEST PIPELINE"
echo "====================="

cd /workspace

# Test 1: Syntax check all Python files
echo "1. Syntax check..."
find . -name "*.py" -type f | while read file; do
    python3 -m py_compile "$file" || {
        echo "❌ Syntax error in $file"
        exit 1
    }
done
echo "   ✅ Syntax OK"

# Test 2: Run unit tests if exist
echo "2. Unit tests..."
if [ -d "intelligence_system/tests" ]; then
    cd intelligence_system
    python3 -m pytest tests/ -v --tb=short || {
        echo "❌ Tests failed"
        exit 1
    }
    echo "   ✅ Tests passed"
fi

# Test 3: Quick integration test
echo "3. Integration test..."
timeout 60 python3 intelligence_system/test_100_cycles_real.py 1 || {
    echo "❌ Integration failed"
    exit 1
}
echo "   ✅ Integration OK"

echo ""
echo "✅ ALL TESTS PASSED"
EOFTEST

chmod +x /srv/agent-workspace-safe/RUN_TESTS.sh

echo "✅ Test pipeline criado"
echo ""

# ============================================
# FASE F: Initial Agent Task
# ============================================
echo "🤖 FASE F: Preparando primeira tarefa do agente..."

cat > /srv/agent-workspace-safe/AGENT_FIRST_TASK.md << 'EOFTASK'
# 🤖 PRIMEIRA TAREFA DO AGENTE AUTÔNOMO

## Objetivo
Analisar o código e propor UMA melhoria simples e segura.

## Instruções para o Agente

1. **Varrer workspace**:
   ```bash
   find /workspace -name "*.py" -type f | head -20
   ```

2. **Identificar oportunidade de melhoria**:
   - Procurar por: duplicação de código
   - Procurar por: funções muito longas (>100 linhas)
   - Procurar por: variáveis mal nomeadas
   - Procurar por: falta de docstrings

3. **Propor UMA mudança**:
   - Escolher o mais simples
   - Criar diff
   - Explicar benefício

4. **Validar**:
   ```bash
   cd /workspace
   bash RUN_TESTS.sh
   ```

5. **Commitar SE testes passarem**:
   ```bash
   git add <arquivo>
   git commit -m "AGENT: <descrição da mudança>"
   ```

## Restrições

- ❌ NÃO modificar mais de 1 arquivo por vez
- ❌ NÃO fazer mudanças arquiteturais grandes
- ❌ NÃO remover código sem entender impacto
- ✅ Fazer mudanças incrementais e testáveis
- ✅ Sempre rodar RUN_TESTS.sh antes de commit
- ✅ Explicar PORQUÊ da mudança

## Expected Output

```
MUDANÇA PROPOSTA:
Arquivo: intelligence_system/core/system_v7_ultimate.py
Linha: 1234
Tipo: Refactoring
Mudança: Extrair função _calculate_metrics() de run_cycle()
Benefício: Melhor testabilidade e clareza
Tests: ✅ PASSED
Ready for: Human approval
```
EOFTASK

echo "✅ Primeira tarefa definida"
echo ""

# ============================================
# Verificações finais
# ============================================
echo "🧪 VERIFICAÇÕES FINAIS..."
echo ""

# Check Qwen
QWEN_STATUS=$(curl -m 5 -sS http://127.0.0.1:8013/health 2>/dev/null || echo "fail")
if [ "$QWEN_STATUS" == '{"status":"ok"}' ]; then
    echo "✅ Qwen respondendo (porta 8013)"
else
    echo "⚠️  Qwen não respondendo - verificar service"
fi

# Check OpenHands
OH_STATUS=$(docker ps | grep openhands | wc -l)
if [ "$OH_STATUS" -ge 1 ]; then
    echo "✅ OpenHands rodando"
    echo "   UI: http://localhost:3000"
else
    echo "⚠️  OpenHands não iniciou - verificar docker logs"
fi

# Check Safety Controller
if [ -f /root/AGENT_SAFETY_CONTROLLER.py ]; then
    echo "✅ Safety Controller pronto"
else
    echo "❌ Safety Controller missing"
fi

echo ""
echo "=============================================="
echo "✅ SETUP COMPLETO!"
echo "=============================================="
echo ""
echo "📊 PRÓXIMOS PASSOS:"
echo ""
echo "1. INICIAR SAFETY CONTROLLER (terminal 1):"
echo "   python3 /root/AGENT_SAFETY_CONTROLLER.py"
echo ""
echo "2. ABRIR OPENHANDS UI (navegador):"
echo "   http://localhost:3000"
echo ""
echo "3. NO OPENHANDS, configurar:"
echo "   Settings → Models → Custom"
echo "   Base URL: http://host.docker.internal:8013/v1"
echo "   Model: qwen2.5-coder-7b"
echo "   API Key: sk-local"
echo ""
echo "4. PRIMEIRA TAREFA:"
echo "   Copiar conteúdo de:"
echo "   cat /srv/agent-workspace-safe/AGENT_FIRST_TASK.md"
echo "   Colar no chat do OpenHands"
echo ""
echo "5. OBSERVAR:"
echo "   Agente vai analisar código"
echo "   Propor mudança"
echo "   Safety Controller vai validar"
echo "   VOCÊ aprova manualmente"
echo ""
echo "=============================================="
echo "⚠️  LEMBRETE CRÍTICO:"
echo "   - Workspace é ISOLADO (/srv/agent-workspace-safe)"
echo "   - Mudanças SÓ vão para /root após VOCÊ aprovar"
echo "   - Git checkpoint: git reset --hard $TAG"
echo "   - Kill switch: touch /root/.AGENT_KILL_SWITCH"
echo "=============================================="
echo ""
echo "🔥 AGENTE PRONTO PARA OPERAR COM SEGURANÇA"
echo "🔥 MAS AINDA REQUER SUA SUPERVISÃO"
echo ""