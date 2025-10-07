#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Fase 3/8 — Aquisição de Conhecimento & Busca Semântica
================================================================
OBJETIVO: Sistema de aquisição inteligente que busca, indexa e recupera conhecimento
relevante para alimentar o pipeline F3→F4→F5→F6, integrando com o sistema multi-API
e fornecendo contexto enriquecido para mutações e auto-evolução.

ENTREGAS:
✓ Worker F3 real integrado ao NEXUS-Ω
✓ Sistema de busca semântica com embeddings
✓ Indexação automática de conhecimento
✓ Integração com sistema multi-API para enriquecimento
✓ Cache inteligente e deduplicação
✓ Métricas de qualidade e relevância

INTEGRAÇÃO SIMBIÓTICA:
- 1/8 (núcleo): recebe OmegaState e atualiza métricas de conhecimento
- 2/8 (estratégia): usa PlanΩ para direcionar busca
- 4/8 (mutação): fornece contexto para geração de candidatos
- 5/8 (crisol): alimenta avaliação com conhecimento relevante
- 6/8 (auto-rewrite): fornece evidências para TTD-DR
- 7/8 (scheduler): registra como worker F3

Autor: Equipe PENIN-Ω
Versão: 3.0.0
"""

from __future__ import annotations
import asyncio
import json
import hashlib
import time
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO E PATHS
# =============================================================================

ROOT = Path("/root/.penin_omega")
ROOT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "KNOWLEDGE": ROOT / "knowledge",
    "EMBEDDINGS": ROOT / "embeddings", 
    "CACHE": ROOT / "cache",
    "LOGS": ROOT / "logs"
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

KNOWLEDGE_DB = DIRS["KNOWLEDGE"] / "knowledge.db"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# INTEGRAÇÃO COM SISTEMA MULTI-API
# =============================================================================

try:
    from penin_omega_fusion_v6 import PeninOmegaFusion
    MULTI_API_AVAILABLE = True
except ImportError:
    MULTI_API_AVAILABLE = False
    logger.info("⚠️  Sistema multi-API não encontrado, usando modo standalone")

# =============================================================================
# MODELOS DE DADOS
# =============================================================================

@dataclass
class KnowledgeItem:
    """Item de conhecimento indexado."""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""

@dataclass
class SearchQuery:
    """Query de busca semântica."""
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    min_similarity: float = 0.3
    boost_recent: bool = True

@dataclass
class AcquisitionResult:
    """Resultado da aquisição F3."""
    query: str
    items: List[KnowledgeItem]
    total_found: int
    processing_time_ms: float
    sources_used: List[str]
    quality_score: float
    enriched_context: str = ""

# =============================================================================
# SISTEMA DE EMBEDDINGS
# =============================================================================

class EmbeddingEngine:
    """Motor de embeddings semânticos."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo de embeddings."""
        try:
            self.model = SentenceTransformer(EMBEDDINGS_MODEL)
            logger.info(f"✅ Modelo de embeddings carregado: {EMBEDDINGS_MODEL}")
        except Exception as e:
            logger.info(f"❌ Erro carregando modelo: {e}")
            self.model = None
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Gera embeddings para texto(s)."""
        if not self.model:
            # Fallback: hash-based pseudo-embedding
            if isinstance(texts, str):
                return [hash(texts) % 1000 / 1000.0] * 384
            return [[hash(t) % 1000 / 1000.0] * 384 for t in texts]
        
        return self.model.encode(texts).tolist()
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calcula similaridade coseno."""
        if not emb1 or not emb2:
            return 0.0
        
        try:
            a, b = np.array(emb1), np.array(emb2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except:
            return 0.0

# =============================================================================
# BASE DE CONHECIMENTO
# =============================================================================

class KnowledgeBase:
    """Base de conhecimento com busca semântica."""
    
    def __init__(self, db_path: Path = KNOWLEDGE_DB):
        self.db_path = db_path
        self.embedding_engine = EmbeddingEngine()
        self._init_db()
    
    def _init_db(self):
        """Inicializa banco de conhecimento."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                embedding TEXT,
                relevance_score REAL DEFAULT 0.0,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_items(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON knowledge_items(relevance_score)")
        conn.commit()
        conn.close()
    
    def add_item(self, content: str, source: str, metadata: Dict[str, Any] = None) -> str:
        """Adiciona item à base de conhecimento."""
        item_id = hashlib.sha256(f"{content}{source}".encode()).hexdigest()[:16]
        embedding = self.embedding_engine.encode(content)
        
        item = KnowledgeItem(
            id=item_id,
            content=content,
            source=source,
            metadata=metadata or {},
            embedding=embedding,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO knowledge_items 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.id, item.content, item.source, 
            json.dumps(item.metadata), json.dumps(item.embedding),
            item.relevance_score, item.created_at, item.updated_at
        ))
        conn.commit()
        conn.close()
        
        return item_id
    
    def search(self, query: SearchQuery) -> List[KnowledgeItem]:
        """Busca semântica na base de conhecimento."""
        query_embedding = self.embedding_engine.encode(query.text)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("""
            SELECT id, content, source, metadata, embedding, relevance_score, created_at, updated_at
            FROM knowledge_items
            ORDER BY relevance_score DESC
            LIMIT ?
        """, (query.limit * 3,))  # Busca mais para filtrar por similaridade
        
        results = []
        for row in cursor.fetchall():
            try:
                embedding = json.loads(row[4]) if row[4] else []
                similarity = self.embedding_engine.similarity(query_embedding, embedding)
                
                if similarity >= query.min_similarity:
                    item = KnowledgeItem(
                        id=row[0],
                        content=row[1],
                        source=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        embedding=embedding,
                        relevance_score=similarity,
                        created_at=row[6],
                        updated_at=row[7]
                    )
                    results.append(item)
            except Exception:
                continue
        
        conn.close()
        
        # Ordena por similaridade e aplica boost temporal se solicitado
        if query.boost_recent:
            now = time.time()
            for item in results:
                try:
                    created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00')).timestamp()
                    age_hours = (now - created) / 3600
                    recency_boost = max(0, 1.0 - age_hours / (24 * 7))  # Boost por 1 semana
                    item.relevance_score = item.relevance_score * (1 + 0.1 * recency_boost)
                except:
                    pass
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:query.limit]

# =============================================================================
# WORKER F3 - AQUISIÇÃO
# =============================================================================

class F3AcquisitionWorker:
    """Worker F3 para aquisição de conhecimento."""
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.multi_api = None
        if MULTI_API_AVAILABLE:
            try:
                self.multi_api = PeninOmegaFusion()
            except:
                pass
    
    async def process_task(self, task_payload: Dict[str, Any]) -> AcquisitionResult:
        """Processa tarefa F3 de aquisição."""
        start_time = time.time()
        
        # Extrai parâmetros da tarefa
        query_text = task_payload.get("query", "")
        goals = task_payload.get("goals", [])
        context = task_payload.get("context", "")
        
        # Constrói query expandida
        expanded_query = self._build_expanded_query(query_text, goals, context)
        
        # Busca na base de conhecimento local
        search_query = SearchQuery(
            text=expanded_query,
            limit=20,
            min_similarity=0.3,
            boost_recent=True
        )
        
        local_items = self.kb.search(search_query)
        
        # Enriquece com sistema multi-API se disponível
        enriched_context = ""
        if self.multi_api and len(local_items) < 5:
            enriched_context = await self._enrich_with_multi_api(expanded_query)
            
            # Adiciona contexto enriquecido à base de conhecimento
            if enriched_context:
                self.kb.add_item(
                    content=enriched_context,
                    source="multi_api_enrichment",
                    metadata={"query": expanded_query, "timestamp": time.time()}
                )
        
        # Calcula métricas de qualidade
        quality_score = self._calculate_quality_score(local_items, enriched_context)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AcquisitionResult(
            query=expanded_query,
            items=local_items,
            total_found=len(local_items),
            processing_time_ms=processing_time,
            sources_used=list(set(item.source for item in local_items)),
            quality_score=quality_score,
            enriched_context=enriched_context
        )
    
    def _build_expanded_query(self, query: str, goals: List[Dict], context: str) -> str:
        """Constrói query expandida com contexto."""
        parts = [query] if query else []
        
        # Adiciona objetivos
        for goal in goals:
            if isinstance(goal, dict) and "name" in goal:
                parts.append(goal["name"])
        
        # Adiciona contexto relevante
        if context:
            # Extrai palavras-chave do contexto
            keywords = re.findall(r'\b[a-zA-Z]{4,}\b', context.lower())
            parts.extend(keywords[:5])  # Top 5 keywords
        
        return " ".join(parts)
    
    async def _enrich_with_multi_api(self, query: str) -> str:
        """Enriquece conhecimento usando sistema multi-API."""
        try:
            from penin_omega_multi_api_integrator import get_global_multi_api_integrator
            
            integrator = get_global_multi_api_integrator()
            if not integrator.is_available():
                return ""
            
            # Usa conector F3 específico
            f3_connector = integrator.get_f3_connector()
            enriched = await f3_connector.enrich_knowledge(query, {
                "goals": [{"name": "enriquecer conhecimento"}],
                "module": "F3"
            })
            
            return enriched
            
        except Exception as e:
            logger.info(f"⚠️  Erro no enriquecimento multi-API: {e}")
            return ""
    
    def _calculate_quality_score(self, items: List[KnowledgeItem], enriched: str) -> float:
        """Calcula score de qualidade da aquisição."""
        if not items and not enriched:
            return 0.0
        
        # Score baseado em quantidade, diversidade e relevância
        quantity_score = min(1.0, len(items) / 10.0)
        
        # Diversidade de fontes
        sources = set(item.source for item in items)
        diversity_score = min(1.0, len(sources) / 5.0)
        
        # Relevância média
        if items:
            avg_relevance = sum(item.relevance_score for item in items) / len(items)
        else:
            avg_relevance = 0.0
        
        # Bonus por enriquecimento
        enrichment_bonus = 0.2 if enriched else 0.0
        
        return min(1.0, (quantity_score + diversity_score + avg_relevance) / 3.0 + enrichment_bonus)

# =============================================================================
# BOOTSTRAP E INDEXAÇÃO
# =============================================================================

def bootstrap_knowledge_base():
    """Inicializa base de conhecimento com dados essenciais."""
    kb = KnowledgeBase()
    
    # Conhecimento básico sobre PENIN-Ω
    essential_knowledge = [
        {
            "content": "PENIN-Ω é um sistema de auto-evolução que usa múltiplas APIs para melhorar continuamente seus algoritmos através de ciclos F3→F4→F5→F6.",
            "source": "system_core",
            "metadata": {"type": "system_overview", "priority": "high"}
        },
        {
            "content": "O pipeline F3 (Aquisição) → F4 (Mutação) → F5 (Crisol) → F6 (Auto-Rewrite) implementa um ciclo completo de evolução algorítmica.",
            "source": "system_core", 
            "metadata": {"type": "pipeline", "priority": "high"}
        },
        {
            "content": "Métricas críticas incluem ρ (risco), SR (success rate), ECE (calibração), PPL (perplexidade) e CAOS⁺ (complexidade).",
            "source": "system_core",
            "metadata": {"type": "metrics", "priority": "high"}
        },
        {
            "content": "Gates de segurança Σ-Guard, IR→IC e SR-Ω∞ garantem que mudanças não degradem performance ou segurança.",
            "source": "system_core",
            "metadata": {"type": "safety", "priority": "critical"}
        }
    ]
    
    for item in essential_knowledge:
        kb.add_item(item["content"], item["source"], item["metadata"])
    
    logger.info(f"✅ Base de conhecimento inicializada com {len(essential_knowledge)} itens essenciais")

# =============================================================================
# API PÚBLICA
# =============================================================================

def create_f3_worker() -> F3AcquisitionWorker:
    """Cria worker F3 para integração com NEXUS-Ω."""
    return F3AcquisitionWorker()

async def f3_acquisition_process(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Função principal para processamento F3."""
    worker = create_f3_worker()
    result = await worker.process_task(task_payload)
    return asdict(result)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main API
    "create_f3_worker", "f3_acquisition_process",
    
    # Core classes
    "F3AcquisitionWorker", "KnowledgeBase", "EmbeddingEngine",
    "KnowledgeItem", "SearchQuery", "AcquisitionResult",
    
    # Utils
    "bootstrap_knowledge_base"
]

if __name__ == "__main__":
    # Teste básico
    logger.info("PENIN-Ω 3/8 - Aquisição de Conhecimento")
    logger.info("Inicializando sistema...")
    
    # Bootstrap da base de conhecimento
    bootstrap_knowledge_base()
    
    # Teste do worker F3
    async def test_f3():
        worker = create_f3_worker()
        result = await worker.process_task({
            "query": "otimização de algoritmos",
            "goals": [{"name": "reduzir perplexidade"}],
            "context": "sistema de auto-evolução"
        })
        logger.info(f"✅ Teste F3 concluído: {result.total_found} itens encontrados")
        logger.info(f"   Quality score: {result.quality_score:.3f}")
        logger.info(f"   Processing time: {result.processing_time_ms:.1f}ms")
    
    import asyncio
    asyncio.run(test_f3())
    logger.info("✅ Código 3/8 funcionando!")
