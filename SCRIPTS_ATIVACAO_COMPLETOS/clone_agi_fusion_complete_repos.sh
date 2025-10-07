#!/usr/bin/env bash
set -euo pipefail

echo "🚀 CLONANDO TODOS OS REPOSITÓRIOS PARA AGI FUSION ULTIMATE"
echo "================================================================"

repos=(
  # Núcleo simbólico / cognitivas
  "https://github.com/opencog/opencog"
  "https://github.com/opencog/atomspace"
  "https://github.com/opennars/opennars"
  "https://github.com/opennars/OpenNARS-for-Applications"
  "https://github.com/SoarGroup/Soar"
  "https://github.com/aimacode/aima-python"
  "https://github.com/pcarbonn/pyDatalog"
  "https://github.com/pyro-ppl/pyro"
  "https://github.com/pyprob/pyprob"

  # Meta-learning / continual / evolução
  "https://github.com/learn2learn/learn2learn"
  "https://github.com/facebookresearch/higher"
  "https://github.com/ContinualAI/avalanche"
  "https://github.com/RobertTLange/evosax"
  "https://github.com/google/evojax"
  "https://github.com/facebookresearch/nevergrad"
  "https://github.com/DEAP/deap"
  "https://github.com/microsoft/nni"
  "https://github.com/optuna/optuna"
  "https://github.com/automl/auto-sklearn"
  "https://github.com/automl/Auto-PyTorch"

  # Auto-modificação / agentes dev
  "https://github.com/All-Hands-AI/OpenHands"
  "https://github.com/princeton-nlp/SWE-agent"
  "https://github.com/princeton-nlp/SWE-bench"
  "https://github.com/smol-ai/developer"
  "https://github.com/AntonOsika/gpt-engineer"
  "https://github.com/OpenInterpreter/open-interpreter"
  "https://github.com/Significant-Gravitas/AutoGPT"
  "https://github.com/yoheinakajima/babyagi"
  "https://github.com/geekan/MetaGPT"

  # Motivação / curiosity
  "https://github.com/pathak22/noreward-rl"
  "https://github.com/openai/large-scale-curiosity"
  "https://github.com/DLR-RM/stable-baselines3"
  "https://github.com/cleanrl/cleanrl"
  "https://github.com/tianshou-org/tianshou"
  "https://github.com/deepmind/acme"

  # Ambientes
  "https://github.com/Farama-Foundation/Gymnasium"
  "https://github.com/Farama-Foundation/PettingZoo"
  "https://github.com/deepmind/meltingpot"
  "https://github.com/openai/procgen"
  "https://github.com/minerllabs/minerl"
  "https://github.com/openai/neural-mmo"
  "https://github.com/google/miniwob-plusplus"
  "https://github.com/deepmind/mujoco"
  "https://github.com/google/brax"
  "https://github.com/facebookresearch/habitat-lab"
  "https://github.com/facebookresearch/habitat-sim"
  "https://github.com/cyberbotics/webots"
  "https://github.com/Unity-Technologies/ml-agents"
  "https://github.com/microsoft/playwright-python"
  "https://github.com/SeleniumHQ/selenium"

  # Modelos de mundo
  "https://github.com/danijar/dreamerv3"
  "https://github.com/danijar/dreamer"

  # Orquestração multi-agente (já existem alguns)
  "https://github.com/microsoft/autogen"
  "https://github.com/langchain-ai/langchain"
  "https://github.com/langchain-ai/langgraph"
  "https://github.com/run-llama/llama_index"
  "https://github.com/joaomdmoura/crewai"
  "https://github.com/microsoft/graphrag"
  "https://github.com/stanfordnlp/dspy"
  "https://github.com/BerriAI/litellm"

  # Memória / RAG / KG
  "https://github.com/facebookresearch/faiss"
  "https://github.com/milvus-io/milvus"
  "https://github.com/weaviate/weaviate"
  "https://github.com/chroma-core/chroma"
  "https://github.com/lancedb/lancedb"
  "https://github.com/cpacker/MemGPT"
  "https://github.com/RDFLib/rdflib"
  "https://github.com/neo4j/neo4j"

  # LLMs / treino / inferência
  "https://github.com/huggingface/transformers"
  "https://github.com/huggingface/trl"
  "https://github.com/CarperAI/trlx"
  "https://github.com/OpenRLHF/OpenRLHF"
  "https://github.com/NVIDIA/Megatron-LM"
  "https://github.com/microsoft/Megatron-DeepSpeed"
  "https://github.com/EleutherAI/gpt-neox"
  "https://github.com/meta-llama/llama"
  "https://github.com/state-spaces/mamba"
  "https://github.com/BlinkDL/RWKV-LM"
  "https://github.com/vllm-project/vllm"
  "https://github.com/huggingface/text-generation-inference"
  "https://github.com/ggerganov/llama.cpp"
  "https://github.com/NVIDIA/TensorRT-LLM"
  "https://github.com/microsoft/DeepSpeed"
  "https://github.com/hpcaitech/ColossalAI"
  "https://github.com/facebookresearch/xformers"
  "https://github.com/Dao-AILab/flash-attention"
  "https://github.com/ray-project/ray"
  "https://github.com/skypilot-org/skypilot"
  "https://github.com/google/jax"
  "https://github.com/google/flax"
  "https://github.com/deepmind/optax"
  "https://github.com/deepmind/dm-haiku"
  "https://github.com/Lightning-AI/pytorch-lightning"
  "https://github.com/mosaicml/composer"

  # Data engine
  "https://github.com/EleutherAI/the-pile"
  "https://github.com/togethercomputer/RedPajama-Data"
  "https://github.com/LAION-AI/laion5B"
  "https://github.com/LAION-AI/Open-Assistant"
  "https://github.com/allenai/dolma"
  "https://github.com/allenai/OLMo"
  "https://github.com/yizhongw/self-instruct"
  "https://github.com/huggingface/datasets"
  "https://github.com/huggingface/tokenizers"

  # Avaliação / segurança
  "https://github.com/EleutherAI/lm-evaluation-harness"
  "https://github.com/stanford-crfm/helm"
  "https://github.com/openai/evals"
  "https://github.com/google/BIG-bench"
  "https://github.com/hendrycks/test"
  "https://github.com/THUDM/AgentBench"
  "https://github.com/NVIDIA/NeMo-Guardrails"
  "https://github.com/guardrails-ai/guardrails"
  "https://github.com/unitaryai/detoxify"
  "https://github.com/pytest-dev/pytest"
  "https://github.com/HypothesisWorks/hypothesis"
  "https://github.com/google/atheris"
  "https://github.com/google/nsjail"
  "https://github.com/netblue30/firejail"
  "https://github.com/open-policy-agent/opa"

  # Observabilidade / MLOps
  "https://github.com/mlflow/mlflow"
  "https://github.com/aimhubio/aim"
  "https://github.com/facebookresearch/hydra"
  "https://github.com/kubernetes/kubernetes"
  "https://github.com/langfuse/langfuse"
  "https://github.com/Arize-ai/phoenix"

  # Código / agentes de programação
  "https://github.com/bigcode-project/starcoder2"
  "https://github.com/facebookresearch/codellama"
  "https://github.com/openai/human-eval"

  # Processamento de dados / ingestão
  "https://github.com/scrapy/scrapy"
  "https://github.com/adbar/trafilatura"
  "https://github.com/codelucas/newspaper"
  "https://github.com/psf/requests"
  "https://github.com/aio-libs/aiohttp"
  "https://github.com/yt-dlp/yt-dlp"

  # Parsing/OCR/ASR
  "https://github.com/Unstructured-IO/unstructured"
  "https://github.com/apache/tika"
  "https://github.com/deanmalmgren/textract"
  "https://github.com/pdfminer/pdfminer.six"
  "https://github.com/jsvine/pdfplumber"
  "https://github.com/pymupdf/PyMuPDF"
  "https://github.com/python-openxml/python-docx"
  "https://github.com/openpyxl/openpyxl"
  "https://github.com/malteos/marker"
  "https://github.com/buriy/python-readability"
  "https://github.com/lxml/lxml"
  "https://github.com/tesseract-ocr/tesseract"
  "https://github.com/PaddlePaddle/PaddleOCR"
  "https://github.com/JaidedAI/EasyOCR"
  "https://github.com/openai/whisper"
  "https://github.com/guillaumekln/faster-whisper"
  "https://github.com/pyannote/pyannote-audio"

  # NLP/Embeddings
  "https://github.com/explosion/spaCy"
  "https://github.com/stanfordnlp/stanza"
  "https://github.com/MaartenGr/BERTopic"
  "https://github.com/RaRe-Technologies/gensim"
  "https://github.com/thunlp/OpenNRE"
  "https://github.com/tree-sitter/tree-sitter"
  "https://github.com/sentence-transformers/sentence-transformers"
  "https://github.com/FlagOpen/FlagEmbedding"
  "https://github.com/huggingface/text-embeddings-inference"
  "https://github.com/stanford-futuredata/ColBERT"
  "https://github.com/BAAI/bge-reranker"

  # Busca/Índices
  "https://github.com/qdrant/qdrant"
  "https://github.com/opensearch-project/OpenSearch"
  "https://github.com/pgvector/pgvector"
  "https://github.com/asg017/sqlite-vss"
  "https://github.com/quickwit-oss/quickwit"

  # ETL/Orquestração
  "https://github.com/apache/airflow"
  "https://github.com/PrefectHQ/prefect"
  "https://github.com/dagster-io/dagster"
  "https://github.com/airbytehq/airbyte"
  "https://github.com/apache/beam"
  "https://github.com/apache/spark"
  "https://github.com/dbt-labs/dbt-core"
  "https://github.com/iterative/dvc"
  "https://github.com/lakefs/lakeFS"
  "https://github.com/pachyderm/pachyderm"

  # Controle OS/Web
  "https://github.com/pyautogui/pyautogui"
  "https://github.com/giampaolo/psutil"
  "https://github.com/pexpect/pexpect"
  "https://github.com/amoffat/sh"
  "https://github.com/fabric/fabric"

  # Agentes específicos não listados ainda
  "https://github.com/huggingface/smolagents"
  "https://github.com/phidatahq/phidata"
  "https://github.com/langroid/langroid"
  "https://github.com/Camel-AI/camel"
  "https://github.com/SuperAGI/superagi"
  "https://github.com/AGiXT/AGiXT"
  "https://github.com/Josh-XT/Agent-LLM"
  "https://github.com/assafelovic/gpt-researcher"
  "https://github.com/openai/swarm"
  "https://github.com/browser-use/browser-use"
  "https://github.com/mem0ai/mem0"

  # Multimodal
  "https://github.com/openai/CLIP"
  "https://github.com/haotian-liu/LLaVA"
  "https://github.com/salesforce/LAVIS"

  # Conhecimento estruturado adicional
  "https://github.com/apache/jena"
  "https://github.com/vaticle/typedb"
  "https://github.com/kuzudb/kuzu"
  "https://github.com/vesoft-inc/nebula"
  "https://github.com/arangodb/arangodb"
  "https://github.com/deepset-ai/haystack"
)

# Criar diretório base para repos se não existir
mkdir -p repos

# Organizar por categoria
categories=(
  "cognitive_symbolic"
  "meta_learning" 
  "auto_modification"
  "motivation_curiosity"
  "environments"
  "world_models"
  "multi_agent"
  "memory_rag"
  "llms_training"
  "data_engine"
  "evaluation_security"
  "observability"
  "code_agents"
  "data_processing"
  "parsing_ocr"
  "nlp_embeddings"
  "search_indices"
  "etl_orchestration"
  "os_web_control"
  "specific_agents"
  "multimodal"
  "knowledge_graphs"
)

# Criar diretórios de categoria
for category in "${categories[@]}"; do
  mkdir -p "repos/$category"
done

echo "📦 Começando a clonar ${#repos[@]} repositórios..."
echo "🗂️  Organizando em ${#categories[@]} categorias"

# Function to categorize repos
categorize_repo() {
  local repo_url="$1"
  local repo_name=$(basename "$repo_url")
  
  case "$repo_name" in
    # Cognitive/Symbolic
    opencog|atomspace|opennars|OpenNARS-for-Applications|Soar|aima-python|pyDatalog|pyro|pyprob)
      echo "cognitive_symbolic";;
    
    # Meta-learning
    learn2learn|higher|avalanche|evosax|evojax|nevergrad|deap|nni|optuna|auto-sklearn|Auto-PyTorch)
      echo "meta_learning";;
    
    # Auto-modification
    OpenHands|SWE-agent|SWE-bench|developer|gpt-engineer|open-interpreter|AutoGPT|babyagi|MetaGPT)
      echo "auto_modification";;
    
    # Motivation/Curiosity
    noreward-rl|large-scale-curiosity|stable-baselines3|cleanrl|tianshou|acme)
      echo "motivation_curiosity";;
    
    # Environments
    Gymnasium|PettingZoo|meltingpot|procgen|minerl|neural-mmo|miniwob-plusplus|mujoco|brax|habitat-lab|habitat-sim|webots|ml-agents|playwright-python|selenium)
      echo "environments";;
    
    # World models
    dreamerv3|dreamer)
      echo "world_models";;
    
    # Multi-agent orchestration
    autogen|langchain|langgraph|llama_index|crewai|graphrag|dspy|litellm|smolagents|phidata|langroid|camel|superagi|AGiXT|Agent-LLM|gpt-researcher|swarm|browser-use)
      echo "multi_agent";;
    
    # Memory/RAG
    faiss|milvus|weaviate|chroma|lancedb|MemGPT|rdflib|neo4j|mem0)
      echo "memory_rag";;
    
    # LLMs/Training
    transformers|trl|trlx|OpenRLHF|Megatron-LM|Megatron-DeepSpeed|gpt-neox|llama|mamba|RWKV-LM|vllm|text-generation-inference|llama.cpp|TensorRT-LLM|DeepSpeed|ColossalAI|xformers|flash-attention|ray|skypilot|jax|flax|optax|dm-haiku|pytorch-lightning|composer)
      echo "llms_training";;
    
    # Data engine
    the-pile|RedPajama-Data|laion5B|Open-Assistant|dolma|OLMo|self-instruct|datasets|tokenizers)
      echo "data_engine";;
    
    # Evaluation/Security
    lm-evaluation-harness|helm|evals|BIG-bench|test|AgentBench|NeMo-Guardrails|guardrails|detoxify|pytest|hypothesis|atheris|nsjail|firejail|opa)
      echo "evaluation_security";;
    
    # Observability
    mlflow|aim|hydra|kubernetes|langfuse|phoenix)
      echo "observability";;
    
    # Code agents
    starcoder2|codellama|human-eval)
      echo "code_agents";;
    
    # Data processing
    scrapy|trafilatura|newspaper|requests|aiohttp|yt-dlp)
      echo "data_processing";;
    
    # Parsing/OCR
    unstructured|tika|textract|pdfminer.six|pdfplumber|PyMuPDF|python-docx|openpyxl|marker|python-readability|lxml|tesseract|PaddleOCR|EasyOCR|whisper|faster-whisper|pyannote-audio)
      echo "parsing_ocr";;
    
    # NLP/Embeddings
    spaCy|stanza|BERTopic|gensim|OpenNRE|tree-sitter|sentence-transformers|FlagEmbedding|text-embeddings-inference|ColBERT|bge-reranker)
      echo "nlp_embeddings";;
    
    # Search/Indices
    qdrant|OpenSearch|pgvector|sqlite-vss|quickwit)
      echo "search_indices";;
    
    # ETL/Orchestration
    airflow|prefect|dagster|airbyte|beam|spark|dbt-core|dvc|lakeFS|pachyderm)
      echo "etl_orchestration";;
    
    # OS/Web control
    pyautogui|psutil|pexpect|sh|fabric)
      echo "os_web_control";;
    
    # Multimodal
    CLIP|LLaVA|LAVIS)
      echo "multimodal";;
    
    # Knowledge graphs
    jena|typedb|kuzu|nebula|arangodb|haystack)
      echo "knowledge_graphs";;
    
    *)
      echo "specific_agents";;
  esac
}

cloned=0
updated=0
failed=0

for url in "${repos[@]}"; do
  name=$(basename "$url")
  category=$(categorize_repo "$url")
  target_dir="repos/$category/$name"
  
  echo -n "📥 $name ($category) ... "
  
  if [[ -d "$target_dir" ]]; then
    # Update existing
    if git -C "$target_dir" pull --ff-only >/dev/null 2>&1; then
      echo "✅ atualizado"
      ((updated++))
    else
      echo "⚠️  falha na atualização"
      ((failed++))
    fi
  else
    # Clone new
    if git clone --depth 1 "$url" "$target_dir" >/dev/null 2>&1; then
      echo "✅ clonado"
      ((cloned++))
    else
      echo "❌ falha no clone"
      ((failed++))
    fi
  fi
done

echo ""
echo "🎉 CLONAGEM COMPLETA!"
echo "✅ Clonados: $cloned"
echo "🔄 Atualizados: $updated"
echo "❌ Falharam: $failed"
echo "📊 Total: $((cloned + updated)) de ${#repos[@]}"

echo ""
echo "📂 Estrutura criada:"
for category in "${categories[@]}"; do
  count=$(find "repos/$category" -maxdepth 1 -type d | wc -l)
  if [[ $count -gt 1 ]]; then
    echo "   📁 $category: $((count - 1)) repositórios"
  fi
done

echo ""
echo "✨ Próximo passo: Integrar tudo no AGI Fusion!"