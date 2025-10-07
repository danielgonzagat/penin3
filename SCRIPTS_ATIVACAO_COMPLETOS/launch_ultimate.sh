#!/usr/bin/env bash
set -euo pipefail

echo "🚀 AGI Fusion Ultimate - Launch Sequence Initiated"
echo "=================================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-flight checks
print_status "Performing pre-flight checks..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# Check git
if ! command -v git &> /dev/null; then
    print_error "Git is not installed"
    exit 1
fi

print_success "Git is available"

# Check if .env exists
if [[ ! -f .env ]]; then
    print_warning "No .env file found. Creating template..."
    cat > .env << 'EOF'
# AGI Fusion Ultimate Configuration
# Add your API keys below

# OpenAI (Required for most features)
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic Claude (Optional)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Google (Optional)
GOOGLE_API_KEY=your-google-api-key-here

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key

# System Configuration
LOG_LEVEL=INFO
DEBUG=false
MAX_WORKERS=4
EOF
    print_warning "Please edit .env file with your API keys before proceeding"
    read -p "Press Enter to continue with limited functionality or Ctrl+C to abort..."
fi

# Setup virtual environment
if [[ ! -d .venv ]]; then
    print_status "Setting up virtual environment..."
    python3 -m venv .venv
fi

print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_status "Installing core dependencies..."
if [[ -f requirements_ultimate.txt ]]; then
    pip install -r requirements_ultimate.txt > /dev/null 2>&1
    print_success "Core dependencies installed"
else
    print_warning "requirements_ultimate.txt not found, installing basic requirements..."
    pip install fastapi uvicorn torch transformers > /dev/null 2>&1
fi

# Install framework packages
print_status "Installing integrated frameworks..."

FRAMEWORKS=(
    "autogen_2:🤖 AutoGen"
    "crewai:👥 CrewAI"
    "dspy:🎯 DSPy"
    "llama_index_2:📚 LlamaIndex"
    "memgpt_2:🧠 MemGPT"
    "langfuse:📊 Langfuse"
)

for framework in "${FRAMEWORKS[@]}"; do
    IFS=':' read -r dir icon_name <<< "$framework"
    name="${icon_name:1}"
    icon="${icon_name:0:1}"

    if [[ -d "$dir" ]]; then
        print_status "$icon Installing $name..."
        cd "$dir"
        if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
            pip install -e . > /dev/null 2>&1 && print_success "$name installed" || print_warning "$name installation failed"
        else
            print_warning "No setup file found for $name"
        fi
        cd ..
    else
        print_warning "$name directory not found"
    fi
done

# Create necessary directories
print_status "Creating system directories..."
mkdir -p memory_store logs checkpoints audit configs

# Test imports
print_status "Testing framework imports..."
python3 -c "
import sys
frameworks = [
    ('torch', 'PyTorch'),
    ('fastapi', 'FastAPI'),
    ('transformers', 'Transformers'),
    ('langchain', 'LangChain'),
]

success_count = 0
for module, name in frameworks:
    try:
        __import__(module)
        print(f'✅ {name}')
        success_count += 1
    except ImportError as e:
        print(f'⚠️  {name}: {e}')

optional_frameworks = [
    ('autogen', 'AutoGen'),
    ('crewai', 'CrewAI'),
    ('dspy', 'DSPy'),
    ('llama_index', 'LlamaIndex'),
    ('memgpt', 'MemGPT'),
]

print('\n📦 Optional Frameworks:')
for module, name in optional_frameworks:
    try:
        __import__(module)
        print(f'✅ {name}')
        success_count += 1
    except ImportError:
        print(f'❌ {name} (not available)')

print(f'\n🎯 Framework Status: {success_count}/{len(frameworks) + len(optional_frameworks)} available')
"

# Test basic functionality
print_status "Testing basic system functionality..."
python3 -c "
from agi_fusion_ultimate import AGIFusionUltimate
print('✅ Core system import successful')

# Test configuration loading
try:
    system = AGIFusionUltimate.__new__(AGIFusionUltimate)
    system.config = system._load_config('config_ultimate.json')
    print('✅ Configuration loading successful')
except Exception as e:
    print(f'⚠️  Configuration loading issue: {e}')
"

# Launch options
echo ""
echo "🎯 AGI Fusion Ultimate Launch Options:"
echo "====================================="
echo ""
echo "1) 🚀 Launch FastAPI Server (Recommended)"
echo "   python app/main_ultimate.py"
echo ""
echo "2) 🧪 Run Integration Tests"
echo "   python test_integration.py"
echo ""
echo "3) 🧬 Start Evolution Engine"
echo "   python auto_evolution_engine.py"
echo ""
echo "4) 📊 View System Status"
echo "   curl http://localhost:8000/agi/status"
echo ""
echo "5) 🎨 Interactive Documentation"
echo "   Visit: http://localhost:8000/docs (after launching server)"
echo ""

# Auto-launch server
read -p "🚀 Launch FastAPI server now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_success "Launching AGI Fusion Ultimate server..."

    # Check if port 8000 is available
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
        print_warning "Port 8000 is already in use. Attempting to free it..."
        fuser -k 8000/tcp || true
        sleep 2
    fi

    # Launch server
    nohup python app/main_ultimate.py > logs/server.log 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > .server_pid

    print_success "Server launched with PID: $SERVER_PID"
    print_success "Logs: logs/server.log"
    print_success "PID file: .server_pid"

    # Wait a moment for server to start
    sleep 3

    # Test server
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "✅ Server is running at http://localhost:8000"
        print_success "🌐 Interactive API docs: http://localhost:8000/docs"
        print_success "📊 System status: http://localhost:8000/agi/status"
    else
        print_warning "⚠️  Server may not have started properly. Check logs/server.log"
    fi

    echo ""
    print_success "🎉 AGI Fusion Ultimate is now LIVE!"
    echo ""
    echo "Try these example API calls:"
    echo "• Health check: curl http://localhost:8000/health"
    echo "• System status: curl http://localhost:8000/agi/status"
    echo "• Test AGI task: curl -X POST http://localhost:8000/agi/execute -H 'Content-Type: application/json' -d '{\"task\":\"Analyze the impact of AGI on society\"}'"

else
    print_status "Server not launched. You can start it manually with:"
    print_status "python app/main_ultimate.py"
fi

echo ""
print_success "🎯 AGI Fusion Ultimate setup complete!"
print_success "Welcome to the future of autonomous AGI systems!"
echo ""
echo "📖 For documentation, see README_ULTIMATE.md"
echo "🧪 To run tests: python test_integration.py"
echo "🔧 To configure: edit .env and config_ultimate.json"
echo ""