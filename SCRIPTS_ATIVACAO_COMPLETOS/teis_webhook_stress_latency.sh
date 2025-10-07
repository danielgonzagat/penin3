#!/usr/bin/env bash
# TEIS Webhook Stress Tester (com latências + CSV)
# Requisitos: bash 4+, curl, awk, date
set -euo pipefail

# ---------- Config ----------
URL="${URL:-http://localhost:8088/webhook/rollback}"   # LOCAL por padrão
SECRET="${SECRET:-change-me}"                           # altere para seu segredo
PAYLOAD='{"title":"Pico de rollbacks (1h)"}'
CONCURRENCY="${CONCURRENCY:-20}"
REQUESTS="${REQUESTS:-200}"
INSECURE="${INSECURE:-false}"                          # false para local
EXTRA_CURL="${EXTRA_CURL:-}"
MODE="${MODE:-ok}"                                      # ok|wrong-secret|no-secret|bad-path
CSV_OUT="${CSV_OUT:-webhook_results.csv}"              # arquivo de saída CSV

# ---------- Helpers ----------
function one_req() {
  local mode="$1" url="$URL"
  local hdr=(-H "Content-Type: application/json")
  case "$mode" in
    ok)            hdr+=(-H "X-TEIS-Secret: $SECRET");;
    wrong-secret)  hdr+=(-H "X-TEIS-Secret: XXXXX-ERRADA");;
    no-secret)     : ;;
    bad-path)      url="${URL%/webhook/rollback}/webhook/rolback"; hdr+=(-H "X-TEIS-Secret: $SECRET");;
    *) echo "modo desconhecido: $mode" >&2; return 2;;
  esac

  local insecure=()
  [[ "$INSECURE" == "true" ]] && insecure=(--insecure)

  # Medir latência com time_total do curl (%{time_total})
  # Exporta: ts_iso,status,time_total
  local ts status ttot
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Usamos --write-out para obter status e tempo total; corpo descartado
  read -r status ttot < <(curl -sS -o /dev/null \
      -w "%{http_code} %{time_total}\n" \
      "${insecure[@]}" "${hdr[@]}" ${EXTRA_CURL} \
      --max-time 15 -X POST "$url" --data "$PAYLOAD")
  echo "$ts,$status,$ttot"
}

export -f one_req
export URL SECRET PAYLOAD INSECURE EXTRA_CURL

# ---------- Execução ----------
echo "Alvo: $URL"
echo "Modo: $MODE | Requests: $REQUESTS | Concurrency: $CONCURRENCY"
echo "CSV:  $CSV_OUT"
echo "timestamp,status,latency_seconds" > "$CSV_OUT"

# Dispara em paralelo e agrega CSV
seq 1 "$REQUESTS" | xargs -I{} -P"$CONCURRENCY" bash -c "one_req '$MODE'" >> "$CSV_OUT"

# ---------- Resumo ----------
echo
echo "=== SUMÁRIO POR STATUS ==="
awk -F, 'NR>1{c[$2]++}END{for (k in c) printf "%s: %d\n", k, c[k]}' "$CSV_OUT" | sort

echo
echo "=== LATÊNCIAS (s) — p50/p90/p95/p99/máx ==="
# Coleta apenas linhas com status 200 e 2xx/4xx/5xx em geral (ajuste se desejar)
awk -F, 'NR>1{print $3}' "$CSV_OUT" | sort -n > .lat_all.txt
# Função awk para percentis
awk '
function pct(arr,n,p,  idx){ idx=int((p/100)*(n-1)+1); if(idx<1)idx=1; if(idx>n)idx=n; return arr[idx]; }
BEGIN{n=0}
{a[++n]=$1}
END{
  if(n==0){ print "sem dados"; exit }
  # já está ordenado; a[] começa em 1
  printf "ALL  p50=%.4f  p90=%.4f  p95=%.4f  p99=%.4f  max=%.4f\n",
    a[int(0.50*(n-1)+1)], a[int(0.90*(n-1)+1)], a[int(0.95*(n-1)+1)], a[int(0.99*(n-1)+1)], a[n]
}' .lat_all.txt

rm -f .lat_all.txt || true

echo
echo "Feito. CSV salvo em: $CSV_OUT"