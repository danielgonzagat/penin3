#!/bin/bash

echo "🚀 Iniciando Interfaces Qwen2.5-Coder-7B..."

# Verifica se o servidor Qwen está rodando
echo "📡 Verificando servidor Qwen..."
if ! curl -s http://127.0.0.1:8013/v1/models >/dev/null; then
    echo "❌ Servidor Qwen não está funcionando!"
    echo "   Inicie o servidor com: sudo systemctl start llama-qwen"
    exit 1
fi

echo "✅ Servidor Qwen está funcionando"

# Menu de opções
echo ""
echo "Escolha uma interface:"
echo "1) Interface Simples (Terminal)"
echo "2) Interface CLI Robusta (Terminal)"
echo "3) Interface Flask (Web - Porta 5000)"
echo "4) Interface Flask Alternativa (Web - Porta 5001)"
echo "5) Interface Gradio (Web Moderna)"
echo "6) Interface Streamlit (Web Responsiva)"
echo "7) Todas as interfaces"
echo ""

read -p "Digite sua escolha (1-7): " choice

case $choice in
    1)
        echo "🚀 Iniciando Interface Simples..."
        python3 /root/qwen_simple_interactive.py
        ;;
    2)
        echo "🚀 Iniciando Interface CLI Robusta..."
        python3 /root/qwen_cli_interface.py
        ;;
    3)
        echo "🚀 Iniciando Interface Flask (Porta 5000)..."
        python3 /root/qwen_flask_interface.py
        ;;
    4)
        echo "🚀 Iniciando Interface Flask Alternativa (Porta 5001)..."
        python3 /root/qwen_flask_interface_alt.py
        ;;
    5)
        echo "🚀 Iniciando Interface Gradio..."
        python3 /root/qwen_gradio_interface.py
        ;;
    6)
        echo "🚀 Iniciando Interface Streamlit..."
        streamlit run /root/qwen_streamlit_interface.py --server.port 8501 --server.address 0.0.0.0
        ;;
    7)
        echo "🚀 Iniciando todas as interfaces..."
        echo "   - Interface Flask: http://localhost:5000"
        echo "   - Interface Gradio: http://localhost:7860"
        echo "   - Interface Streamlit: http://localhost:8501"
        echo ""
        
        # Inicia Flask em background
        python3 /root/qwen_flask_interface.py &
        FLASK_PID=$!
        
        # Inicia Gradio em background
        python3 /root/qwen_gradio_interface.py &
        GRADIO_PID=$!
        
        # Inicia Streamlit em background
        streamlit run /root/qwen_streamlit_interface.py --server.port 8501 --server.address 0.0.0.0 &
        STREAMLIT_PID=$!
        
        echo "✅ Todas as interfaces iniciadas!"
        echo "   PIDs: Flask=$FLASK_PID, Gradio=$GRADIO_PID, Streamlit=$STREAMLIT_PID"
        echo ""
        echo "Pressione Ctrl+C para parar todas as interfaces..."
        
        # Aguarda Ctrl+C
        trap "kill $FLASK_PID $GRADIO_PID $STREAMLIT_PID 2>/dev/null; echo '👋 Interfaces encerradas'; exit 0" INT
        wait
        ;;
    *)
        echo "❌ Opção inválida!"
        exit 1
        ;;
esac
