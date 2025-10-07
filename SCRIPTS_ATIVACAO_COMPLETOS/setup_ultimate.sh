#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up AGI Fusion Ultimate System"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📋 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [[ ! -d .venv ]]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements_ultimate.txt

# Install framework-specific dependencies
echo "🔧 Installing framework dependencies..."

# AutoGen
if [[ -d autogen_2 ]]; then
    echo "🤖 Installing AutoGen..."
    cd autogen_2
    pip install -e .
    cd ..
fi

# CrewAI
if [[ -d crewai ]]; then
    echo "👥 Installing CrewAI..."
    cd crewai
    pip install -e .
    cd ..
fi

# DSPy
if [[ -d dspy ]]; then
    echo "🎯 Installing DSPy..."
    cd dspy
    pip install -e .
    cd ..
fi

# LlamaIndex
if [[ -d llama_index_2 ]]; then
    echo "📚 Installing LlamaIndex..."
    cd llama_index_2
    pip install -e .
    cd ..
fi

# MemGPT
if [[ -d memgpt_2 ]]; then
    echo "🧠 Installing MemGPT..."
    cd memgpt_2
    pip install -e .
    cd ..
fi

# Langfuse
if [[ -d langfuse ]]; then
    echo "📊 Installing Langfuse..."
    cd langfuse
    pip install -e .
    cd ..
fi

# vLLM (optional, requires GPU)
if [[ -d vllm ]] && command -v nvidia-smi &> /dev/null; then
    echo "⚡ Installing vLLM..."
    cd vllm
    pip install -e .
    cd ..
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p memory_store
mkdir -p logs
mkdir -p configs
mkdir -p checkpoints
mkdir -p audit

# Setup configuration
echo "⚙️  Setting up configuration..."
if [[ ! -f .env ]]; then
    cat > .env << EOF
# AGI Fusion Ultimate Configuration
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Database
DATABASE_URL=sqlite:///agi_fusion.db

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key

# Redis (Optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key

# System
MAX_WORKERS=4
LOG_LEVEL=INFO
DEBUG=false
EOF
    echo "✅ Created .env file - please edit with your API keys"
fi

# Initialize memory stores
echo "🧠 Initializing memory systems..."
python3 -c "
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# Create empty index for LlamaIndex
try:
    documents = []
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir='./memory_store/llamaindex')
    print('✅ LlamaIndex memory initialized')
except Exception as e:
    print(f'⚠️  LlamaIndex initialization skipped: {e}')
"

# Test imports
echo "🧪 Testing imports..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)

    import langgraph
    print('✅ LangGraph available')

    import autogen
    print('✅ AutoGen available')

    import crewai
    print('✅ CrewAI available')

    import dspy
    print('✅ DSPy available')

    from llama_index.core import VectorStoreIndex
    print('✅ LlamaIndex available')

    import memgpt
    print('✅ MemGPT available')

    print('🎉 All core frameworks imported successfully!')

except ImportError as e:
    print(f'⚠️  Import warning: {e}')
"

# Create systemd service (optional)
if [[ "${INSTALL_SERVICE:-false}" == "true" ]]; then
    echo "🔧 Creating systemd service..."
    sudo tee /etc/systemd/system/agi-fusion.service > /dev/null << EOF
[Unit]
Description=AGI Fusion Ultimate System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$SCRIPT_DIR/.venv/bin
ExecStart=$SCRIPT_DIR/.venv/bin/python -m uvicorn app.main_ultimate:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable agi-fusion
    echo "✅ Service created - run 'sudo systemctl start agi-fusion' to start"
fi

echo ""
echo "🎯 AGI Fusion Ultimate Setup Complete!"
echo "====================================="
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python app/main_ultimate.py"
echo "3. Visit: http://localhost:8000/docs"
echo ""
echo "🔗 Available frameworks:"
echo "   • LangGraph: Multi-agent orchestration"
echo "   • AutoGen: Conversational AI agents"
echo "   • CrewAI: Team-based agent coordination"
echo "   • DSPy: Automatic prompt optimization"
echo "   • LlamaIndex: Advanced RAG and memory"
echo "   • MemGPT: Long-term memory management"
echo "   • Langfuse: LLM observability"
echo ""
echo "🚀 Ready for autonomous AGI operations!"