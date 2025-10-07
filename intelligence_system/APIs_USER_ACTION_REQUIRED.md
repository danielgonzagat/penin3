# 🔑 APIs Requiring User Action

**Status**: 4/6 APIs funcionando (Mistral, DeepSeek, Grok, Gemini-flash)  
**Ação Necessária**: Resolver 2-3 APIs para 100% coverage

---

## 🔴 OpenAI GPT-5

**Status Atual**: 401 Unauthorized  
**Fallback Ativo**: ✅ `gpt-4.1-2025-04-14` (automático)

**Ação Necessária**:
1. **Verificar API Key**: https://platform.openai.com/settings/organization/api-keys
2. **Verificar Billing**: https://platform.openai.com/settings/organization/billing
3. **Confirmar Acesso GPT-5**: Modelo pode requerer waitlist/preview access
4. **Se GPT-5 indisponível**: Sistema continua com GPT-4.1 (fallback robusto)

**Código Atual**:
```python
# /root/intelligence_system/apis/litellm_wrapper.py:145-197
# Já implementado: Responses API com fallback automático para GPT-4.1
```

**Urgência**: ⚠️ MÉDIA (fallback funciona perfeitamente)

---

## 🔴 Gemini 2.5-Pro

**Status Atual**: Missing API key / API not enabled  
**Fallback Ativo**: ✅ `gemini-1.5-flash` (automático)

**Ação Necessária**:
1. **Habilitar Generative Language API**:
   - https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
   - Clicar em "ENABLE"
   
2. **Verificar API Key**:
   - https://makersuite.google.com/app/apikey
   - Confirmar key `AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k` está ativa
   
3. **Verificar Quotas e Billing**:
   - https://console.cloud.google.com/iam-admin/quotas
   - https://console.cloud.google.com/billing

4. **Exportar variável de ambiente**:
   ```bash
   export GEMINI_API_KEY='AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k'
   export GOOGLE_API_KEY="$GEMINI_API_KEY"
   ```

**Código Atual**:
```python
# /root/intelligence_system/apis/litellm_wrapper.py:200-228
# Já implementado: google-genai SDK com fallback para 1.5-flash
```

**Urgência**: ⚠️ MÉDIA (fallback funciona perfeitamente)

---

## 🔴 Anthropic Opus 4.1

**Status Atual**: 401 Authentication Error  
**Fallback Ativo**: ✅ `claude-3-5-sonnet-20240620` (automático)

**Ação Necessária**:
1. **Verificar API Key**:
   - https://console.anthropic.com/settings/keys
   - Confirmar key `sk-ant-api03-jnm8q5nLOh...` está ativa
   - Se expirada, gerar nova key

2. **Verificar Acesso ao Modelo**:
   - Opus 4.1 (`claude-opus-4-1-20250805`) pode requerer plano específico
   - Verificar em https://console.anthropic.com/settings/plans
   
3. **Testar Key Manualmente**:
   ```bash
   pip install anthropic
   python3 -c "
   import anthropic
   client = anthropic.Anthropic(api_key='sk-ant-api03-...')
   msg = client.messages.create(
       model='claude-3-5-sonnet-20240620',
       max_tokens=16,
       messages=[{'role': 'user', 'content': 'ping'}]
   )
   print(msg.content[0].text)
   "
   ```

4. **Se 401 persistir**: Gerar nova key ou usar Sonnet (fallback)

**Código Atual**:
```python
# /root/intelligence_system/apis/litellm_wrapper.py:283-310
# Já implementado: anthropic SDK com fallback para Sonnet/Haiku
```

**Urgência**: ⚠️ MÉDIA (fallback funciona perfeitamente)

---

## ℹ️ DeepSeek Reasoner

**Status Atual**: Indisponível (deepseek-chat OK)  
**Fallback Ativo**: ✅ `deepseek-chat` (mode não-pensante)

**Ação Necessária** (opcional):
1. **Verificar Account Status**:
   - https://platform.deepseek.com/api_keys
   - Confirmar key `sk-19c2b1d0864c4a44a53d743fb97566aa` tem acesso

2. **Request Beta Access** (se reasoner desejado):
   - https://api-docs.deepseek.com/guides/reasoning_model
   - Solicitar acesso ao thinking mode

3. **Alternativa**: Usar `deepseek-chat` (atual, funcionando)

**Código Atual**:
```python
# /root/intelligence_system/apis/litellm_wrapper.py:230-281
# Já implementado: OpenAI client com base_url
# Suporta chat e reasoner (quando disponível)
```

**Urgência**: ℹ️ BAIXA (chat mode suficiente para uso geral)

---

## ✅ APIs Funcionando Perfeitamente

### Mistral Codestral-2508
**Status**: ✅ OK  
**Key**: `AMTeAQrzudpGvU2jkU9hVRvSsYr1hcni`  
**Código**: `apis/litellm_wrapper.py` (LiteLLM padrão)

### DeepSeek Chat
**Status**: ✅ OK  
**Key**: `sk-19c2b1d0864c4a44a53d743fb97566aa`  
**Código**: `apis/litellm_wrapper.py:230-281` (OpenAI client)

### Grok-4
**Status**: ✅ OK  
**Key**: `xai-sHbr1x7v2vpfDi657DtU64U53UM6OVhs4FdHeR1Ijk7jRUgU0xmo6ff8SF7hzV9mzY1wwjo4ChYsCDog`  
**Código**: `apis/litellm_wrapper.py` (LiteLLM padrão)

### Gemini 1.5-Flash (fallback)
**Status**: ✅ OK  
**Key**: `AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k`  
**Código**: `apis/litellm_wrapper.py:200-228` (google-genai SDK)

---

## 🎯 Quick Test: Verificar Status APIs

```bash
cd /root/intelligence_system

# Exportar env vars (usar suas keys reais)
export OPENAI_API_KEY='sk-proj-4JrC7R3cl_...'
export MISTRAL_API_KEY='AMTeAQrzudpGvU2jkU9hVRvSsYr1hcni'
export GEMINI_API_KEY='AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k'
export GOOGLE_API_KEY="$GEMINI_API_KEY"
export DEEPSEEK_API_KEY='sk-19c2b1d0864c4a44a53d743fb97566aa'
export ANTHROPIC_API_KEY='sk-ant-api03-jnm8q5nLOh...'
export GROK_API_KEY='xai-sHbr1x7v2vpfDi657DtU64U53UM6OVhs4FdHeR1Ijk7jRUgU0xmo6ff8SF7hzV9mzY1wwjo4ChYsCDog'

# Teste rápido
python3 - << 'PY'
import os, json
from intelligence_system.apis.litellm_wrapper import LiteLLMWrapper
from intelligence_system.config.settings import API_KEYS, API_MODELS

wrapper = LiteLLMWrapper(API_KEYS, API_MODELS)
results = wrapper.call_all_models_robust('ping', max_tokens=16)

print(json.dumps({
    'total': len(API_MODELS),
    'ok': len(results),
    'providers': {k: ('✅ OK' if k in results else '❌ FAIL') for k in API_MODELS.keys()}
}, indent=2))
PY
```

**Esperado**: 4-5/6 APIs OK após fallbacks

---

**Última Atualização**: 03 Outubro 2025, 21:18 UTC  
**Próxima Ação**: Resolver APIs faltantes ou aceitar fallbacks
