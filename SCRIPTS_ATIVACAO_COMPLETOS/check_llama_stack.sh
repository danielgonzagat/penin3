#!/usr/bin/env bash
# /root/check_llama_stack.sh
set -euo pipefail

API_KEY="${API_KEY:-DANIEL}"
MODEL="${MODEL:-qwen2.5-7b-instruct}"
HOST="127.0.0.1"
NGINX_PORT=8080
B0=8090
B1=8091

_red()  { printf "\e[31m%s\e[0m\n" "$*"; }
_grn()  { printf "\e[32m%s\e[0m\n" "$*"; }
_yel()  { printf "\e[33m%s\e[0m\n" "$*"; }
_hdr()  { printf "\n==== %s ====\n" "$*"; }
_die()  { _red "ERRO: $*"; exit 1; }

_check_cmd() {
  command -v "$1" >/dev/null 2>&1 || _die "comando '$1' não encontrado"
}

_json() {
  local URL="$1"
  local DATA="$2"
  curl -sS "$URL" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$DATA"
}

_get() {
  local URL="$1"
  curl -sS "$URL" -H "Authorization: Bearer ${API_KEY}"
}

_hdr "Pré-checagens"
_check_cmd curl
_check_cmd systemctl
_check_cmd ss
if ! systemctl is-active --quiet nginx; then _yel "nginx não está active (ok se usando apenas backends)"; fi

_hdr "Status dos serviços (llama-s0 / llama-s1)"
systemctl --no-pager --full -n 3 status llama-s0 || true
systemctl --no-pager --full -n 3 status llama-s1 || true
systemctl is-active --quiet llama-s0 || _die "llama-s0 não está active"
systemctl is-active --quiet llama-s1 || _die "llama-s1 não está active"
_grn "OK: ambos serviços estão active"

_hdr "Portas escutando"
ss -ltn | awk 'NR==1 || /:8090|:8091|:8080/'
ss -ltn | grep -q ":${B0} " || _die "porta ${B0} não escutando"
ss -ltn | grep -q ":${B1} " || _die "porta ${B1} não escutando"
if ! ss -ltn | grep -q ":${NGINX_PORT} "; then
  _yel "porta ${NGINX_PORT} (nginx) não está escutando — balanceamento pode estar desligado"
else
  _grn "OK: ${NGINX_PORT} está escutando"
fi
_grn "OK: backends ouvindo"

_hdr "Testes /v1/models (direto nos backends)"
OUT_B0="$(_get "http://${HOST}:${B0}/v1/models" | head -c 200 || true)"
OUT_B1="$(_get "http://${HOST}:${B1}/v1/models" | head -c 200 || true)"
[[ "$OUT_B0" == *"models"* ]] || _die "backend ${B0} não respondeu como esperado"
[[ "$OUT_B1" == *"models"* ]] || _die "backend ${B1} não respondeu como esperado"
_grn "OK: /v1/models responde em ${B0} e ${B1}"

if ss -ltn | grep -q ":${NGINX_PORT} "; then
  _hdr "Teste /v1/models via Nginx (${NGINX_PORT})"
  OUT_NGX="$(_get "http://${HOST}:${NGINX_PORT}/v1/models" | head -c 200 || true)"
  [[ "$OUT_NGX" == *"models"* ]] || _die "/v1/models via ${NGINX_PORT} falhou (veja nginx -T e logs)"
  _grn "OK: /v1/models via ${NGINX_PORT}"
fi

_hdr "Teste /v1/chat/completions real (via ${NGINX_PORT} se disponível, senão ${B0})"
URL="http://${HOST}:${NGINX_PORT}/v1/chat/completions"
if ! ss -ltn | grep -q ":${NGINX_PORT} "; then
  URL="http://${HOST}:${B0}/v1/chat/completions"
  _yel "usando backend direto (${B0}) pois ${NGINX_PORT} não está ativo"
fi

REQ=$(cat <<JSON
{"model":"${MODEL}","messages":[{"role":"user","content":"Diga 'pong'."}],"max_tokens":16}
JSON
)
RESP="$(_json "$URL" "$REQ" || true)"
echo "$RESP" | head -c 300
echo
[[ "$RESP" == *"choices"* ]] || _die "chat/completions não retornou objeto válido"
[[ "$RESP" == *"pong"* ]] && _grn "OK: resposta contém 'pong'"

_hdr "Balanceamento (se log de upstream existir)"
if [[ -f /var/log/nginx/access_upstream.log ]]; then
  # gera 6 reqs para observar alternância
  for i in {1..6}; do
    _get "http://${HOST}:${NGINX_PORT}/v1/models" >/dev/null 2>&1 || true
  done
  tail -n 20 /var/log/nginx/access_upstream.log || true
  if tail -n 50 /var/log/nginx/access_upstream.log | grep -E "127\.0\.0\.1:(${B0}|${B1})" >/dev/null 2>&1; then
    _grn "OK: log mostra upstream_addr (verifique alternância entre ${B0}/${B1})"
  else
    _yel "não encontrei upstream_addr nos últimos logs — confira formato de log do Nginx"
  fi
else
  _yel "arquivo /var/log/nginx/access_upstream.log não existe (balanceamento ainda pode estar ok)"
fi

_hdr "Resumo final"
_grn "✓ serviços ativos (llama-s0/llama-s1)"
_grn "✓ portas ${B0}/${B1} ok"
if ss -ltn | grep -q ":${NGINX_PORT} "; then _grn "✓ Nginx ${NGINX_PORT} ok"; else _yel "• Nginx ${NGINX_PORT} ausente"; fi
_grn "✓ /v1/models ok (backends e, se ativo, via Nginx)"
_grn "✓ /v1/chat/completions ok"

exit 0
