#!/usr/bin/env bash
set -euo pipefail

echo "🚀 INICIANDO CLONAGEM PARA EVOLUÇÃO DO AGI FUSION"
echo "📦 Total estimado: 140+ repositórios"
echo "⏱️  Tempo estimado: 30-60 minutos"
echo ""

# Criar estrutura de diretórios
mkdir -p repos/{frameworks,automation,memory,rag,knowledge,reasoning,agents,meta_learning,creativity,evolution,sensors,vision,robotics,iot,symbolic,causal,planning,infrastructure,serving_llm,evaluation,security,guardrails,observability,mlops,code,agents_programming,simulation,environments,world_models,training_llm,data_engine,pipelines,research,benchmarks}

cd repos

# 1. FRAMEWORKS INTEGRADOS (Núcleo Atual) - PRIORIDADE ALTA
echo "📦 Clonando Frameworks Integrados (14 repos)..."
frameworks=(
    "https://github.com/stanfordnlp/dspy"
    "https://github.com/huggingface/trl"
    "https://github.com/cpacker/MemGPT"
    "https://github.com/microsoft/autogen"
    "https://github.com/joaomdmoura/crewai"
    "https://github.com/langchain-ai/langgraph"
    "https://github.com/OpenInterpreter/open-interpreter"
    "https://github.com/All-Hands-AI/OpenHands"
    "https://github.com/browser-use/browser-use"
    "https://github.com/run-llama/llama_index"
    "https://github.com/BerriAI/litellm"
    "https://github.com/openai/swarm"
    "https://github.com/geekan/MetaGPT"
    "https://github.com/smol-ai/developer"
)

for url in "${frameworks[@]}"; do
    name=$(basename "$url")
    if [[ -d "frameworks/$name" ]]; then
        echo "  ↻ Atualizando $name..."
        git -C "frameworks/$name" pull --ff-only || true
    else
        echo "  ⬇️  Clonando $name..."
        git clone --depth 1 "$url" "frameworks/$name" || true
    fi
done

# 2. EXECUÇÃO DE CÓDIGO, AUTOMAÇÃO E CONTROLE - PRIORIDADE ALTA
echo ""
echo "🤖 Clonando Automação e Controle (7 repos)..."
automation=(
    "https://github.com/microsoft/playwright-python"
    "https://github.com/SeleniumHQ/selenium"
    "https://github.com/pyautogui/pyautogui"
    "https://github.com/giampaolo/psutil"
    "https://github.com/pexpect/pexpect"
    "https://github.com/fabric/fabric"
    "https://github.com/amoffat/sh"
)

for url in "${automation[@]}"; do
    name=$(basename "$url")
    if [[ -d "automation/$name" ]]; then
        git -C "automation/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "automation/$name" || true
    fi
done

# 3. MEMÓRIA, RAG E CONHECIMENTO - PRIORIDADE ALTA
echo ""
echo "🧠 Clonando Memória e RAG (7 repos)..."
memory_rag=(
    "https://github.com/facebookresearch/faiss"
    "https://github.com/milvus-io/milvus"
    "https://github.com/weaviate/weaviate"
    "https://github.com/chroma-core/chroma"
    "https://github.com/lancedb/lancedb"
    "https://github.com/RDFLib/rdflib"
    "https://github.com/neo4j/neo4j"
)

for url in "${memory_rag[@]}"; do
    name=$(basename "$url")
    if [[ -d "memory/$name" ]]; then
        git -C "memory/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "memory/$name" || true
    fi
done

# 4. META-LEARNING E RLHF - PRIORIDADE MÉDIA
echo ""
echo "🧬 Clonando Meta-Learning (8 repos)..."
meta_learning=(
    "https://github.com/huggingface/peft"
    "https://github.com/microsoft/LoRA"
    "https://github.com/artidoro/qlora"
    "https://github.com/CarperAI/trlx"
    "https://github.com/OpenRLHF/OpenRLHF"
    "https://github.com/ContinualAI/avalanche"
    "https://github.com/learn2learn/learn2learn"
    "https://github.com/facebookresearch/higher"
)

for url in "${meta_learning[@]}"; do
    name=$(basename "$url")
    if [[ -d "meta_learning/$name" ]]; then
        git -C "meta_learning/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "meta_learning/$name" || true
    fi
done

# 5. CRIATIVIDADE E EVOLUÇÃO - PRIORIDADE MÉDIA
echo ""
echo "🎨 Clonando Criatividade e Evolução (6 repos)..."
creativity=(
    "https://github.com/CodeReclaimers/neat-python"
    "https://github.com/DEAP/deap"
    "https://github.com/facebookresearch/nevergrad"
    "https://github.com/google/evojax"
    "https://github.com/RobertTLange/evosax"
    "https://github.com/eriklindernoren/PyTorch-GAN"
)

for url in "${creativity[@]}"; do
    name=$(basename "$url")
    if [[ -d "creativity/$name" ]]; then
        git -C "creativity/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "creativity/$name" || true
    fi
done

# 6. SENSORES, VISÃO E ROBÓTICA - PRIORIDADE MÉDIA
echo ""
echo "👁️  Clonando Sensores e Visão (7 repos)..."
sensors=(
    "https://github.com/opencv/opencv"
    "https://github.com/IntelRealSense/librealsense"
    "https://github.com/luxonis/depthai-python"
    "https://github.com/luxonis/depthai"
    "https://github.com/eclipse/mosquitto"
    "https://github.com/eclipse/paho.mqtt.python"
    "https://github.com/deepmind/mujoco"
)

for url in "${sensors[@]}"; do
    name=$(basename "$url")
    if [[ -d "sensors/$name" ]]; then
        git -C "sensors/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "sensors/$name" || true
    fi
done

# 7. SIMBÓLICO E CAUSAL - PRIORIDADE ALTA
echo ""
echo "🧠 Clonando Simbólico e Causal (6 repos)..."
symbolic=(
    "https://github.com/opencog/atomspace"
    "https://github.com/opennars/opennars"
    "https://github.com/pcarbonn/pyDatalog"
    "https://github.com/py-why/dowhy"
    "https://github.com/microsoft/EconML"
    "https://github.com/deepmind/open_spiel"
)

for url in "${symbolic[@]}"; do
    name=$(basename "$url")
    if [[ -d "symbolic/$name" ]]; then
        git -C "symbolic/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "symbolic/$name" || true
    fi
done

# 8. INFRAESTRUTURA E SERVING LLM - PRIORIDADE ALTA
echo ""
echo "🏗️  Clonando Infraestrutura (8 repos)..."
infrastructure=(
    "https://github.com/ray-project/ray"
    "https://github.com/vllm-project/vllm"
    "https://github.com/huggingface/text-generation-inference"
    "https://github.com/ggerganov/llama.cpp"
    "https://github.com/microsoft/DeepSpeed"
    "https://github.com/hpcaitech/ColossalAI"
    "https://github.com/facebookresearch/xformers"
    "https://github.com/Dao-AILab/flash-attention"
)

for url in "${infrastructure[@]}"; do
    name=$(basename "$url")
    if [[ -d "infrastructure/$name" ]]; then
        git -C "infrastructure/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "infrastructure/$name" || true
    fi
done

# 9. SEGURANÇA E GUARDRAILS - PRIORIDADE ALTA
echo ""
echo "🛡️  Clonando Segurança (6 repos)..."
security=(
    "https://github.com/NVIDIA/NeMo-Guardrails"
    "https://github.com/guardrails-ai/guardrails"
    "https://github.com/unitaryai/detoxify"
    "https://github.com/ELEUTHERAI/lm-evaluation-harness"
    "https://github.com/stanford-crfm/helm"
    "https://github.com/THUDM/AgentBench"
)

for url in "${security[@]}"; do
    name=$(basename "$url")
    if [[ -d "security/$name" ]]; then
        git -C "security/$name" pull --ff-only || true
    else
        git clone --depth 1 "$url" "security/$name" || true
    fi
done

echo ""
echo "✅ CLONAGEM CONCLUÍDA!"
echo ""
echo "📊 RESUMO:"
echo "   Frameworks: $(find frameworks -name ".git" | wc -l) repos"
echo "   Automação: $(find automation -name ".git" | wc -l) repos"
echo "   Memória/RAG: $(find memory -name ".git" | wc -l) repos"
echo "   Meta-Learning: $(find meta_learning -name ".git" | wc -l) repos"
echo "   Criatividade: $(find creativity -name ".git" | wc -l) repos"
echo "   Sensores: $(find sensors -name ".git" | wc -l) repos"
echo "   Simbólico: $(find symbolic -name ".git" | wc -l) repos"
echo "   Infraestrutura: $(find infrastructure -name ".git" | wc -l) repos"
echo "   Segurança: $(find security -name ".git" | wc -l) repos"
echo ""
echo "🎯 PRÓXIMO PASSO: Integrar no AGI Fusion!"
echo "📁 Localização: /root/repos/"