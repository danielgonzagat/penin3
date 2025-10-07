#!/bin/bash
set -e

echo "🚀 Iniciando upgrade completo da ETΩ..."

# 1. Instalar dependências do painel
echo "📦 Instalando dependências do painel web..."
pip install fastapi uvicorn plotly jinja2 sympy --quiet

# 2. Criar diretório do painel
mkdir -p /opt/et_ultimate/web

# 3. Criar painel monitor.py
cat <<EOF >/opt/et_ultimate/web/monitor.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import json, os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def show_graph():
    path = "/opt/et_ultimate/history/etomega_scores.jsonl"
    if not os.path.exists(path): return "<h1>Sem dados ainda</h1>"
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    fig = go.Figure()
    ias = list(set(x['ia'] for x in data))
    for ia in ias:
        y = [x['score'] for x in data if x['ia'] == ia]
        fig.add_trace(go.Scatter(y=y, mode='lines+markers', name=ia))
    return fig.to_html()
EOF

# 4. Criar serviço systemd para o painel
cat <<EOF >/etc/systemd/system/etomega-monitor.service
[Unit]
Description=Painel Web ETΩ
After=network.target

[Service]
ExecStart=/usr/bin/uvicorn monitor:app --reload --port 6060
WorkingDirectory=/opt/et_ultimate/web
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

# 5. Ativar painel
systemctl daemon-reload
systemctl enable etomega-monitor
systemctl start etomega-monitor

echo "🌐 Painel disponível em http://<IP>:6060"

# 6. Ajustar frequência do Liga Copilotos para cada 30s
echo "🔁 Aumentando frequência de execução da Liga de Copilotos..."
sed -i 's/ExecStart=.*/ExecStart=\/usr\/bin\/python3 \/opt\/et_ultimate\/agents\/copilots\/et_liga_copilotos.py --interval 30/' /etc/systemd/system/et_liga_copilotos.service || true
systemctl daemon-reexec
systemctl daemon-reload
systemctl restart et_liga_copilotos

# 7. Criar benchmark científico em Python
echo "🧪 Adicionando função evaluate com SymPy..."
BENCH_PATH="/opt/et_ultimate/agents/tools/scientific_eval.py"
cat <<EOF >"$BENCH_PATH"
from sympy import simplify

def scientific_score(eq_str):
    try:
        simplified = simplify(eq_str)
        return 100 - len(str(simplified))
    except Exception:
        return 0
EOF

# 8. Gerar token do nó para rede P2P
echo "🌍 Gerando token para nó ETΩ P2P..."
mkdir -p /opt/et_ultimate/secrets
uuidgen > /opt/et_ultimate/secrets/etomega_node.token

echo "✅ Upgrade completo!"
