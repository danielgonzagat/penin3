# API Troubleshooting Guide

## Status Atual: 4/6 APIs Funcionando ✅

### ✅ APIs Operacionais
1. **Mistral** - codestral-2508
2. **Gemini** - gemini-2.5-pro (com fallback para 1.5-pro/flash)
3. **Anthropic** - claude-opus-4-1-20250805 (com fallback para sonnet/haiku)
4. **Grok** - grok-4

### ⚠️ APIs com Problemas

#### OpenAI (401 Authentication Error)
**Sintoma**:
```
OpenAI Responses API error: 401
```

**Causas Possíveis**:
1. Chave API expirada
2. Chave API revogada
3. Quota/billing issue na conta OpenAI
4. Endpoint incorreto (verificar se /v1/responses está disponível)

**Solução**:
1. Gerar nova chave em: https://platform.openai.com/api-keys
2. Atualizar no `.env`:
   ```bash
   OPENAI_API_KEY=sk-proj-NOVA_CHAVE_AQUI
   ```
3. Testar diretamente:
   ```bash
   curl https://api.openai.com/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4.1-2025-04-14","messages":[{"role":"user","content":"OK"}]}'
   ```

**Workaround Temporário**:
- Sistema continua funcionando com 5 APIs
- Fallback automático para outras APIs em `call_all_models_robust()`

---

#### DeepSeek (401 Authentication Error)
**Sintoma**:
```
Error code: 401 - Authentication Fails, Your api key: ****66aa is invalid
```

**Causas Possíveis**:
1. Chave API incorreta (typo no copy/paste)
2. Chave API revogada ou expirada
3. Account suspended/billing issue

**Solução**:
1. Verificar chave em: https://platform.deepseek.com/api_keys
2. Gerar nova chave se necessário
3. Atualizar no `.env`:
   ```bash
   DEEPSEEK_API_KEY=sk-NOVA_CHAVE_AQUI
   ```
4. Testar diretamente:
   ```bash
   curl https://api.deepseek.com/chat/completions \
     -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"OK"}]}'
   ```

**Nota sobre DeepSeek-V3.1**:
- Modelo atual: `deepseek-chat` (non-thinking)
- Modelo thinking: `deepseek-reasoner` (ativar com `DEEPSEEK_REASONER=1`)
- Endpoint beta: https://api.deepseek.com/beta (para strict function calling)

**Workaround Temporário**:
- Sistema continua funcionando com 5 APIs
- DeepSeek é opcional para operação do sistema

---

## Validação Rápida de Todas as APIs

### Smoke Test Completo
```bash
cd /root
python3 intelligence_system/tools/smoke_apis.py
```

**Resultado Esperado**:
```json
{
  "mistral": "OK.",
  "gemini": "OK.",
  "anthropic": "OK",
  "grok": "OK.",
  "openai": "...",
  "deepseek": "..."
}
```

### Teste Individual (DeepSeek)
```python
import requests
resp = requests.post(
    'https://api.deepseek.com/chat/completions',
    headers={'Authorization': 'Bearer sk-SUA_CHAVE_AQUI'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'OK'}]}
)
print(resp.json())
```

### Teste Individual (OpenAI)
```python
import requests
resp = requests.post(
    'https://api.openai.com/v1/responses',
    headers={'Authorization': 'Bearer sk-proj-SUA_CHAVE_AQUI'},
    json={'model': 'gpt-5', 'input': 'OK'}
)
print(resp.json())
```

---

## Health Dashboard

### Gerar Dashboard
```bash
python3 intelligence_system/tools/health_exporter.py
```

### Visualizar
```bash
# HTML
open intelligence_system/data/exports/api_health.html

# CSV
cat intelligence_system/data/exports/api_health.csv
```

### Dados Inclusos
- Success rate por provider
- Latência média
- Contadores de erro
- Último erro registrado

---

## Logs de Debug

### Ver chamadas API no sistema
```bash
tail -f intelligence_system/data/unified_agi.log | grep -E "(API|LiteLLM|call_model)"
```

### Ver health updates
```bash
watch -n 5 cat intelligence_system/data/api_health.json
```

### Ver circuit breaker status
```python
from intelligence_system.apis.litellm_wrapper import get_provider_status_summary
print(get_provider_status_summary())
```

---

## Fallback Chain Configurado

### OpenAI
```
gpt-5 → gpt-4.1-2025-04-14
```

### Gemini
```
gemini-2.5-pro → gemini-1.5-pro → gemini-1.5-flash
```

### Anthropic
```
claude-opus-4-1-20250805 → claude-3-5-sonnet-20240620 → claude-3-haiku-20240307
```

### DeepSeek
```
deepseek-chat (padrão)
deepseek-reasoner (se DEEPSEEK_REASONER=1)
```

---

## Checklist de Validação

- [ ] Todas as chaves configuradas no `.env`
- [ ] Smoke test executado: `python3 intelligence_system/tools/smoke_apis.py`
- [ ] Health dashboard gerado e verificado
- [ ] Pelo menos 4/6 APIs respondendo
- [ ] Fallback chains testados
- [ ] Circuit breaker funcionando (providers com falhas ficam em cooldown)
- [ ] Health metrics sendo persistidos em `api_health.json`

---

## Support

Se problemas persistirem:
1. Verifique billing/quota em cada plataforma
2. Teste chaves diretamente via curl/requests
3. Confirme endpoints corretos (alguns mudaram recentemente)
4. Verifique rate limits e retry logic
5. Considere proxy/VPN se houver blocking geográfico