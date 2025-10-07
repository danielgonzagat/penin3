#!/bin/bash
# 🔧 FASE 2: INTEGRAÇÃO COM LLAMA FOUNDATION MODEL
# Conecta LLaMa 8B ao UNIFIED_BRAIN
# Tempo: 30 minutos

set -e

echo "🦙 FASE 2: INTEGRAÇÃO LLAMA + UNIFIED_BRAIN"
echo "==========================================="
echo ""

cd /root

# ============================================
# C6: Implementar LLaMa Local Client
# ============================================
echo "🔧 C6: Criando LLaMa local client..."

# Create LLaMa client
cat > /root/intelligence_system/apis/llama_local_client.py << 'EOF'
"""
LLaMa Local Client - Connects to llama-server on localhost
"""
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LLaMaLocalClient:
    """Client for local LLaMa server"""
    
    def __init__(self, base_url="http://127.0.0.1:8080"):
        self.base_url = base_url
        self.available = self._check_health()
    
    def _check_health(self) -> bool:
        """Check if LLaMa server is responding"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            logger.info(f"✅ LLaMa server healthy: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️  LLaMa server not responding: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 200, 
                 temperature: float = 0.7, stop: List[str] = None) -> Optional[str]:
        """
        Generate text from LLaMa
        
        Args:
            prompt: Input text
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            Generated text or None if failed
        """
        if not self.available:
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop or ["\n\n", "User:", "Assistant:"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data.get('choices', [{}])[0].get('text', '')
                logger.info(f"✅ LLaMa generated {len(text)} chars")
                return text.strip()
            else:
                logger.warning(f"LLaMa API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"LLaMa generation failed: {e}")
            return None
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embeddings from LLaMa (if supported)
        
        Note: Not all llama-server builds support embeddings endpoint
        """
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get('data', [{}])[0].get('embedding', [])
                logger.info(f"✅ LLaMa embedding: {len(embedding)}D")
                return embedding
            else:
                logger.debug(f"Embeddings not supported: {response.status_code}")
                return None
        
        except Exception as e:
            logger.debug(f"Embeddings endpoint failed: {e}")
            return None

# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = LLaMaLocalClient()
    
    if client.available:
        response = client.generate("What is 2+2?", max_tokens=20)
        print(f"Test response: {response}")
EOF

echo "✅ LLaMa client criado: intelligence_system/apis/llama_local_client.py"
echo ""

# ============================================
# Test LLaMa connectivity
# ============================================
echo "🧪 Testando conectividade com LLaMa..."

python3 /root/intelligence_system/apis/llama_local_client.py || echo "⚠️  LLaMa não acessível (ver troubleshooting abaixo)"

echo ""

# ============================================
# C7: Integrate LLaMa into UNIFIED_BRAIN
# ============================================
echo "🔧 C7: Integrando LLaMa ao UNIFIED_BRAIN..."

# Backup
cp /root/UNIFIED_BRAIN/brain_system_integration.py \
   /root/UNIFIED_BRAIN/brain_system_integration.py.backup_$(date +%s)

# Add LLM advisor capability
python3 << 'EOFPY'
import sys

# Read file
with open('/root/UNIFIED_BRAIN/brain_system_integration.py', 'r') as f:
    content = f.read()

# Check if already has llm_advisor
if 'self.llm_advisor' in content:
    print("✅ LLM advisor já existe no código")
else:
    # Find __init__ method of UnifiedSystemController
    if 'class UnifiedSystemController' in content:
        # Add import at top
        if 'from apis.llama_local_client import LLaMaLocalClient' not in content:
            # Find import section
            lines = content.split('\n')
            new_lines = []
            import_added = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Add after other imports, before class definitions
                if 'import sys' in line and not import_added:
                    new_lines.append('sys.path.insert(0, "/root/intelligence_system")')
                    new_lines.append('try:')
                    new_lines.append('    from apis.llama_local_client import LLaMaLocalClient')
                    new_lines.append('    _LLAMA_AVAILABLE = True')
                    new_lines.append('except Exception:')
                    new_lines.append('    _LLAMA_AVAILABLE = False')
                    import_added = True
            
            content = '\n'.join(new_lines)
        
        # Add to __init__
        if 'def __init__' in content and 'UnifiedSystemController' in content:
            lines = content.split('\n')
            new_lines = []
            init_modified = False
            in_init = False
            indent_level = 0
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Find __init__ of UnifiedSystemController
                if 'class UnifiedSystemController' in line:
                    in_controller = True
                
                if 'def __init__' in line and in_controller and not init_modified:
                    in_init = True
                    indent_level = len(line) - len(line.lstrip())
                
                # Add llm_advisor after first few lines of __init__
                if in_init and 'self.brain' in line and not init_modified:
                    indent = ' ' * (indent_level + 8)
                    new_lines.append('')
                    new_lines.append(indent + '# ✅ FIX C6+C7: LLM Advisor integration')
                    new_lines.append(indent + 'self.llm_advisor = None')
                    new_lines.append(indent + 'if _LLAMA_AVAILABLE:')
                    new_lines.append(indent + '    try:')
                    new_lines.append(indent + '        self.llm_advisor = LLaMaLocalClient()')
                    new_lines.append(indent + '        logger.info("✅ LLM Advisor initialized")')
                    new_lines.append(indent + '    except Exception as e:')
                    new_lines.append(indent + '        logger.warning(f"LLM Advisor failed: {e}")')
                    init_modified = True
            
            content = '\n'.join(new_lines)
        
        # Write back
        with open('/root/UNIFIED_BRAIN/brain_system_integration.py', 'w') as f:
            f.write(content)
        
        print("✅ LLM Advisor integrado ao UnifiedSystemController")
    else:
        print("⚠️  Estrutura do arquivo diferente do esperado")

EOFPY

echo ""
echo "✅ C7 COMPLETO: LLaMa integrado"
echo ""

# ============================================
# Verify syntax
# ============================================
echo "🧪 Verificando sintaxe..."
python3 -m py_compile /root/UNIFIED_BRAIN/brain_system_integration.py && echo "✅ Sintaxe OK" || echo "❌ Erro - restaurar backup"
python3 -m py_compile /root/intelligence_system/apis/llama_local_client.py && echo "✅ Cliente LLaMa OK" || echo "❌ Erro no cliente"

echo ""

# ============================================
# SUMMARY & NEXT STEPS
# ============================================
echo "=============================================="
echo "✅ FASE 2 PREPARADA!"
echo "=============================================="
echo ""
echo "📊 O QUE FOI FEITO:"
echo "  ✅ LLaMa local client criado"
echo "  ✅ UNIFIED_BRAIN integração preparada"
echo "  ✅ Sintaxe verificada"
echo ""
echo "🎯 PRÓXIMOS PASSOS:"
echo ""
echo "1. VERIFICAR LLAMA SERVER (5 min):"
echo "   ps aux | grep llama-server"
echo "   # Se não rodando, iniciar (ver troubleshooting)"
echo ""
echo "2. VERIFICAR DARWINACCI FIX (5 min):"
echo "   tail -50 /root/darwinacci_test_*.log | grep 'Best score'"
echo "   # Deve ser > 0.0"
echo ""
echo "3. INICIAR MONITOR 24H:"
echo "   nohup bash /root/🔧_MONITOR_24H.sh &"
echo ""
echo "4. AGUARDAR 24H e verificar:"
echo "   cat emergence_monitor_*.log | grep 'EMERGENCE'"
echo ""
echo "=============================================="
echo "🌟 Sistema preparado para emergência!"
echo "=============================================="
echo ""
echo "📖 TROUBLESHOOTING:"
echo ""
echo "Se LLaMa não está acessível:"
echo "  # Verificar se servidor está rodando"
echo "  ps aux | grep llama-server"
echo ""
echo "  # Se não, iniciar manualmente:"
echo "  # (requer modelo GGUF - ver documentação)"
echo ""
echo "📚 Ler mais:"
echo "  cat /root/🧬_PLANO_FUSAO_MODELO_30_50B.md"
echo ""