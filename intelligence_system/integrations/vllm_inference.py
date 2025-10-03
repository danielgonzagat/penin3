"""
vLLM Integration - COMPLETO - Fast LLM Inference
Merge REAL do /root/vllm com funcionalidade completa
"""
import logging
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import from installed vllm
try:
    sys.path.insert(0, '/root/vllm')
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
    logger.info("✅ vLLM imported successfully from /root/vllm")
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")
    VLLM_AVAILABLE = False

class VLLMInference:
    """
    vLLM-based fast inference engine - PRODUCTION READY
    Ultra-fast inference for local models with PagedAttention
    """
    
    def __init__(self, model_name: Optional[str] = None, 
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None):
        self.vllm_available = VLLM_AVAILABLE
        self.model_name = model_name
        self.llm = None
        self.generation_count = 0
        
        if VLLM_AVAILABLE and model_name:
            self._init_vllm(gpu_memory_utilization, max_model_len)
        
        logger.info(f"🚀 vLLM Inference initialized (available: {VLLM_AVAILABLE})")
    
    def _init_vllm(self, gpu_memory_utilization: float, max_model_len: Optional[int]):
        """Initialize vLLM engine with optimizations"""
        try:
            init_kwargs = {
                'model': self.model_name,
                'gpu_memory_utilization': gpu_memory_utilization,
                'trust_remote_code': True
            }
            
            if max_model_len:
                init_kwargs['max_model_len'] = max_model_len
            
            self.llm = LLM(**init_kwargs)
            logger.info(f"✅ vLLM loaded model: {self.model_name}")
            logger.info(f"   GPU memory utilization: {gpu_memory_utilization}")
        except Exception as e:
            logger.error(f"vLLM initialization failed: {e}")
            self.vllm_available = False
    
    def generate(self, prompts: List[str], 
                max_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = -1,
                repetition_penalty: float = 1.0) -> List[str]:
        """
        Generate text with vLLM (ultra-fast with PagedAttention)
        
        Args:
            prompts: List of prompts
            max_tokens: Max generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling (-1 = disabled)
            repetition_penalty: Repetition penalty
        
        Returns:
            List of generated texts
        """
        if not self.vllm_available or self.llm is None:
            logger.warning("vLLM not available, returning empty results")
            return ["[vLLM not available]"] * len(prompts)
        
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            
            self.generation_count += len(prompts)
            
            results = [output.outputs[0].text for output in outputs]
            logger.info(f"✅ vLLM generated {len(results)} completions")
            
            return results
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return [f"[Error: {str(e)}]"] * len(prompts)
    
    def generate_single(self, prompt: str, **kwargs) -> str:
        """Generate single completion (convenience method)"""
        results = self.generate([prompt], **kwargs)
        return results[0] if results else "[Error]"
    
    def batch_generate(self, prompts: List[str], batch_size: int = 32, **kwargs) -> List[str]:
        """
        Generate in batches for very large prompt lists
        
        Args:
            prompts: List of prompts
            batch_size: Batch size for generation
            **kwargs: Generation parameters
        
        Returns:
            List of all generated texts
        """
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            results = self.generate(batch, **kwargs)
            all_results.extend(results)
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive vLLM statistics"""
        stats = {
            'vllm_available': self.vllm_available,
            'model_name': self.model_name,
            'initialized': self.llm is not None,
            'generation_count': self.generation_count
        }
        
        if self.llm:
            stats['engine_status'] = 'ready'
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if vLLM is ready for inference"""
        return self.vllm_available and self.llm is not None

