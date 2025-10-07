class LocalLLMProvider(AIProvider):
    """Provider para Falcon Mamba 7B via HTTP (porta 8010)."""

    def __init__(self, base_url: str = "http://localhost:8010"):
        self._name = "falcon_mamba_7b"
        self.base_url = base_url.rstrip("/")
        self.session = None

    @property
    def name(self) -> str:
        return self._name

    async def execute(self,
                      prompt: str,
                      system_prompt: str = "",
                      **kwargs) -> AIResponse:
        """Executa geração usando Falcon Mamba via HTTP."""
        start_time = time.time()

        if not HAS_AIOHTTP:
            return AIResponse(
                self.name,
                "ERROR",
                error="aiohttp não disponível"
            )

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Preparar payload para o Falcon Mamba
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 0.9)
            }

            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extrair resposta do formato OpenAI-compatible
                    content = ""
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0].get("message", {}).get("content", "")
                    elif "response" in result:
                        content = result["response"]
                    elif "output" in result:
                        content = result["output"]
                    
                    return AIResponse(
                        provider=self.name,
                        status="COMPLETED",
                        content=content,
                        latency=time.time() - start_time
                    )
                else:
                    error_text = await response.text()
                    return AIResponse(
                        provider=self.name,
                        status="ERROR",
                        error=f"HTTP {response.status}: {error_text}",
                        latency=time.time() - start_time
                    )

        except Exception as e:
            return AIResponse(
                provider=self.name,
                status="ERROR",
                error=str(e),
                latency=time.time() - start_time
            )

    async def validate_connection(self) -> bool:
        """Valida conexão com Falcon Mamba."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def close(self):
        """Fecha sessão HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
