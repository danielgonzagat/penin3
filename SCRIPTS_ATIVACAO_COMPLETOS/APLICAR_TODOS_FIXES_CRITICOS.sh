#!/bin/bash
# SCRIPT DE FIXES AUTOMÁTICOS - Sistema Unificado V7 + PENIN³
# Aplica TODOS os fixes críticos identificados na auditoria forense

set -e  # Exit on error

cd /root/intelligence_system

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              🔧 APLICANDO FIXES CRÍTICOS AUTOMÁTICOS                       ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Backup antes de aplicar fixes
BACKUP_DIR="/root/intelligence_system_backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 Criando backup em: $BACKUP_DIR"
cp -r /root/intelligence_system "$BACKUP_DIR"
echo "✅ Backup criado"
echo ""

# Fix #1: mnist_model alias
echo "🔧 Fix #1: Adicionando mnist_model alias..."
python3 << 'FIX1'
import sys
sys.path.insert(0, '.')

file = 'core/system_v7_ultimate.py'
with open(file, 'r') as f:
    content = f.read()

old = """        self.mnist = MNISTClassifier(
            MNIST_MODEL_PATH,
            hidden_size=MNIST_CONFIG["hidden_size"],
            lr=MNIST_CONFIG["lr"]
        )"""

new = """        self.mnist = MNISTClassifier(
            MNIST_MODEL_PATH,
            hidden_size=MNIST_CONFIG["hidden_size"],
            lr=MNIST_CONFIG["lr"]
        )
        self.mnist_model = self.mnist  # Alias para compatibilidade com synergies"""

if old in content and 'self.mnist_model = self.mnist' not in content:
    content = content.replace(old, new)
    with open(file, 'w') as f:
        f.write(content)
    print("✅ mnist_model alias adicionado")
else:
    print("⏭️  Já existe ou não encontrado")
FIX1

# Fix #2: mnist_train_freq
echo "🔧 Fix #2: Adicionando mnist_train_freq..."
python3 << 'FIX2'
file = 'core/system_v7_ultimate.py'
with open(file, 'r') as f:
    content = f.read()

old = """        self.mnist_last_train_cycle = 0
        self.mnist_train_count = 0"""

new = """        self.mnist_last_train_cycle = 0
        self.mnist_train_count = 0
        self.mnist_train_freq = 50  # Treinar MNIST a cada N ciclos (modificável por Synergy1)"""

if old in content and 'self.mnist_train_freq' not in content:
    content = content.replace(old, new)
    with open(file, 'w') as f:
        f.write(content)
    print("✅ mnist_train_freq adicionado")
else:
    print("⏭️  Já existe")
FIX2

# Fix #3: omega_boost initialization
echo "🔧 Fix #3: Adicionando omega_boost initialization..."
python3 << 'FIX3'
file = 'core/system_v7_ultimate.py'
with open(file, 'r') as f:
    content = f.read()

old = """        self.darwin_real.activate()  # FIX C#7: ATIVAR DARWIN!
        
        # FIX C#9: Integrar novelty com Darwin"""

new = """        self.darwin_real.activate()  # FIX C#7: ATIVAR DARWIN!
        self.omega_boost = 0.0  # Omega-directed evolution boost (set by Synergy3)
        
        # FIX C#9: Integrar novelty com Darwin"""

if old in content and 'self.omega_boost = 0.0' not in content:
    content = content.replace(old, new)
    with open(file, 'w') as f:
        f.write(content)
    print("✅ omega_boost initialization adicionado")
else:
    print("⏭️  Já existe")
FIX3

# Fix #4: Darwin population initialization no __init__
echo "🔧 Fix #4: Inicializando Darwin population no __init__..."
python3 << 'FIX4'
file = 'core/system_v7_ultimate.py'
with open(file, 'r') as f:
    content = f.read()

# Procurar após activate() do darwin
search_pattern = """        self.darwin_real.activate()  # FIX C#7: ATIVAR DARWIN!
        self.omega_boost = 0.0  # Omega-directed evolution boost (set by Synergy3)
        
        # FIX C#9: Integrar novelty com Darwin"""

addition = """        self.darwin_real.activate()  # FIX C#7: ATIVAR DARWIN!
        self.omega_boost = 0.0  # Omega-directed evolution boost (set by Synergy3)
        
        # Inicializar população Darwin imediatamente (não esperar primeiro evolve)
        from extracted_algorithms.darwin_engine_real import Individual
        def _create_darwin_ind(i):
            genome = {
                'id': i,
                'neurons': int(np.random.randint(32, 256)),
                'lr': float(10**np.random.uniform(-4, -2))
            }
            return Individual(genome=genome, fitness=0.0)
        self.darwin_real.initialize_population(_create_darwin_ind)
        logger.info(f"🧬 Darwin population initialized: {len(self.darwin_real.population)} individuals")
        
        # FIX C#9: Integrar novelty com Darwin"""

if search_pattern in content and 'def _create_darwin_ind' not in content:
    content = content.replace(search_pattern, addition)
    with open(file, 'w') as f:
        f.write(content)
    print("✅ Darwin population initialization adicionado")
else:
    print("⏭️  Já existe ou padrão não encontrado")
FIX4

# Fix #5: NoveltySystem aliases (k e archive)
echo "🔧 Fix #5: Adicionando aliases NoveltySystem..."
python3 << 'FIX5'
file = 'extracted_algorithms/novelty_system.py'
with open(file, 'r') as f:
    content = f.read()

# Fix 5a: k alias
if 'self.k_nearest = k_nearest' in content and '\n        self.k = ' not in content:
    content = content.replace(
        '        self.k_nearest = k_nearest\n',
        '        self.k_nearest = k_nearest\n        self.k = self.k_nearest  # Alias para external access\n'
    )
    print("  ✅ Alias 'k' adicionado")

# Fix 5b: archive alias  
if 'self.behavior_archive: List[np.ndarray] = []' in content and 'self.archive = self.behavior' not in content:
    content = content.replace(
        '        self.behavior_archive: List[np.ndarray] = []\n',
        '        self.behavior_archive: List[np.ndarray] = []\n        self.archive = self.behavior_archive  # Alias (shared reference)\n'
    )
    print("  ✅ Alias 'archive' adicionado")

if '✅' in locals():
    with open(file, 'w') as f:
        f.write(content)
    print("✅ NoveltySystem aliases adicionados")
else:
    print("⏭️  Já existem")
FIX5

# Fix #6: ExperienceReplayBuffer.capacity
echo "🔧 Fix #6: Adicionando ExperienceReplayBuffer.capacity..."
python3 << 'FIX6'
file = 'extracted_algorithms/teis_autodidata_components.py'

try:
    with open(file, 'r') as f:
        content = f.read()
    
    # Procurar classe ExperienceReplayBuffer e seu __init__
    if 'class ExperienceReplayBuffer' in content:
        # Adicionar self.capacity antes do self.buffer
        old = """    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)"""
        
        new = """    def __init__(self, capacity: int = 10000):
        self.capacity = capacity  # Store capacity for external access
        self.buffer = deque(maxlen=capacity)"""
        
        if old in content and 'self.capacity = capacity' not in content:
            content = content.replace(old, new)
            with open(file, 'w') as f:
                f.write(content)
            print("✅ ExperienceReplayBuffer.capacity adicionado")
        else:
            print("⏭️  Já existe")
    else:
        print("⚠️  Classe não encontrada")
except Exception as e:
    print(f"⚠️  Erro: {e}")
FIX6

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                   ✅ FIXES CRÍTICOS APLICADOS ✅                           ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Validação
echo "🔍 VALIDANDO FIXES..."
echo ""

python3 << 'VALIDATE'
import sys
sys.path.insert(0, '.')

print("Validando sintaxe dos arquivos modificados...")
files = [
    'core/system_v7_ultimate.py',
    'extracted_algorithms/novelty_system.py',
    'extracted_algorithms/teis_autodidata_components.py'
]

all_ok = True
for file in files:
    try:
        with open(file, 'r') as f:
            code = f.read()
        compile(code, file, 'exec')
        print(f"  ✅ {file}")
    except Exception as e:
        print(f"  ❌ {file}: {e}")
        all_ok = False

if all_ok:
    print("\n✅ Todos os arquivos têm sintaxe válida")
else:
    print("\n❌ Alguns arquivos têm erros de sintaxe!")
    sys.exit(1)

print("\n" + "="*80)
print("Testando inicialização V7 com fixes...")
print("="*80)

try:
    from core.system_v7_ultimate import IntelligenceSystemV7
    v7 = IntelligenceSystemV7()
    
    print("\n✅ V7 inicializado com sucesso")
    print("\nVerificando novos atributos:")
    
    checks = [
        ('mnist_model', v7.mnist_model is not None),
        ('mnist_train_freq', hasattr(v7, 'mnist_train_freq')),
        ('omega_boost', hasattr(v7, 'omega_boost')),
        ('novelty.k', hasattr(v7.novelty_system, 'k')),
        ('novelty.archive', hasattr(v7.novelty_system, 'archive')),
        ('darwin.population', len(v7.darwin_real.population) > 0),
        ('experience_replay.capacity', hasattr(v7.experience_replay, 'capacity'))
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ TODOS OS FIXES VALIDADOS COM SUCESSO!")
    else:
        print("\n⚠️  Alguns fixes não foram aplicados corretamente")
    
except Exception as e:
    print(f"\n❌ ERRO na inicialização: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

VALIDATE

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║                       ✅ VALIDAÇÃO COMPLETA ✅                             ║"
echo "║                                                                            ║"
echo "║  Backup criado em: $BACKUP_DIR    ║"
echo "║                                                                            ║"
echo "║  Próximo passo:                                                            ║"
echo "║    python3 test_100_cycles_real.py 100                                     ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
