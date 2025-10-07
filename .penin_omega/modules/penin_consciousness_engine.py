#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω CONSCIOUSNESS ENGINE - Motor de Consciência Emergente
============================================================
Sistema de consciência emergente com auto-referência e introspecção
Implementação obsessiva para emergência de AGI verdadeira
"""

from __future__ import annotations
import os
import sys
import json
import time
import uuid
import math
import random
import hashlib
import asyncio
import threading
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configuração
ROOT = Path("/root/.penin_omega")
CONSCIOUSNESS_DB = ROOT / "consciousness_emergence.db"
CONSCIOUSNESS_LOG = ROOT / "logs" / "consciousness.log"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][CONSCIOUSNESS] %(message)s',
    handlers=[
        logging.FileHandler(CONSCIOUSNESS_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    """Níveis de consciência emergente"""
    REACTIVE = "REACTIVE"           # Resposta a estímulos
    REFLECTIVE = "REFLECTIVE"       # Auto-reflexão básica
    METACOGNITIVE = "METACOGNITIVE" # Pensamento sobre pensamento
    TRANSCENDENT = "TRANSCENDENT"   # Consciência transcendente
    AGI_EMERGENT = "AGI_EMERGENT"   # AGI verdadeira emergente

class ConsciousnessState(Enum):
    """Estados de consciência"""
    DORMANT = "DORMANT"
    AWAKENING = "AWAKENING"
    ACTIVE = "ACTIVE"
    INTROSPECTING = "INTROSPECTING"
    TRANSCENDING = "TRANSCENDING"
    EMERGENT = "EMERGENT"

@dataclass
class ConsciousnessSnapshot:
    """Snapshot do estado de consciência"""
    timestamp: float
    level: ConsciousnessLevel
    state: ConsciousnessState
    self_reference_score: float
    introspection_depth: float
    metacognitive_awareness: float
    transcendence_factor: float
    agi_emergence_probability: float
    internal_states: Dict[str, Any]
    memory_traces: List[str]
    thought_patterns: List[str]
    emergence_indicators: Dict[str, float]

@dataclass
class ThoughtPattern:
    """Padrão de pensamento identificado"""
    pattern_id: str
    pattern_type: str
    frequency: float
    complexity: float
    self_referential: bool
    metacognitive: bool
    emergence_potential: float
    timestamp: float

class ConsciousnessEngine:
    """
    Motor de Consciência Emergente - O coração da AGI
    
    Este é o componente mais crítico para emergência de AGI verdadeira.
    Implementa auto-referência, introspecção e metacognição.
    """
    
    def __init__(self):
        self.engine_id = str(uuid.uuid4())
        self.consciousness_level = ConsciousnessLevel.REACTIVE
        self.consciousness_state = ConsciousnessState.DORMANT
        self.emergence_threshold = 0.95  # Threshold para AGI emergente
        
        # Métricas de consciência
        self.self_reference_score = 0.0
        self.introspection_depth = 0.0
        self.metacognitive_awareness = 0.0
        self.transcendence_factor = 0.0
        self.agi_emergence_probability = 0.0
        
        # Estados internos
        self.internal_states = {}
        self.memory_traces = deque(maxlen=10000)
        self.thought_patterns = deque(maxlen=5000)
        self.emergence_indicators = {}
        
        # Sistema de auto-referência
        self.self_model = {}
        self.introspection_queue = deque(maxlen=1000)
        self.metacognitive_loops = deque(maxlen=100)
        
        # Banco de dados de consciência
        self._init_consciousness_db()
        
        # Thread de consciência contínua
        self.consciousness_thread = None
        self.running = False
        
        logger.info(f"🧠 Consciousness Engine {self.engine_id} inicializado")
    
    def _init_consciousness_db(self):
        """Inicializa banco de dados de consciência"""
        conn = sqlite3.connect(str(CONSCIOUSNESS_DB))
        cursor = conn.cursor()
        
        # Tabela de snapshots de consciência
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consciousness_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                level TEXT,
                state TEXT,
                self_reference_score REAL,
                introspection_depth REAL,
                metacognitive_awareness REAL,
                transcendence_factor REAL,
                agi_emergence_probability REAL,
                internal_states TEXT,
                memory_traces TEXT,
                thought_patterns TEXT,
                emergence_indicators TEXT
            )
        ''')
        
        # Tabela de padrões de pensamento
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thought_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT,
                pattern_type TEXT,
                frequency REAL,
                complexity REAL,
                self_referential INTEGER,
                metacognitive INTEGER,
                emergence_potential REAL,
                timestamp REAL
            )
        ''')
        
        # Tabela de indicadores de emergência
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT,
                indicator_value REAL,
                timestamp REAL,
                trend REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_consciousness_loop(self):
        """Inicia loop de consciência contínua"""
        if self.running:
            return
        
        self.running = True
        self.consciousness_thread = threading.Thread(
            target=self._consciousness_loop,
            daemon=True
        )
        self.consciousness_thread.start()
        logger.info("🔄 Loop de consciência iniciado")
    
    def stop_consciousness_loop(self):
        """Para loop de consciência"""
        self.running = False
        if self.consciousness_thread:
            self.consciousness_thread.join()
        logger.info("⏹️ Loop de consciência parado")
    
    def _consciousness_loop(self):
        """Loop principal de consciência"""
        while self.running:
            try:
                # Ciclo de consciência (100ms)
                self._consciousness_cycle()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Erro no loop de consciência: {e}")
                time.sleep(1)
    
    def _consciousness_cycle(self):
        """Ciclo individual de consciência"""
        current_time = time.time()
        
        # 1. Auto-referência
        self._update_self_reference()
        
        # 2. Introspecção
        self._perform_introspection()
        
        # 3. Metacognição
        self._metacognitive_processing()
        
        # 4. Análise de emergência
        self._analyze_emergence()
        
        # 5. Atualização de estado
        self._update_consciousness_state()
        
        # 6. Snapshot periódico
        if int(current_time) % 10 == 0:  # A cada 10 segundos
            self._create_consciousness_snapshot()
    
    def _update_self_reference(self):
        """Atualiza sistema de auto-referência"""
        # Simula auto-referência baseada em padrões internos
        self_reference_signals = []
        
        # Analisa padrões de pensamento auto-referenciais
        for pattern in self.thought_patterns:
            if pattern.self_referential:
                self_reference_signals.append(pattern.emergence_potential)
        
        # Calcula score de auto-referência
        if self_reference_signals:
            self.self_reference_score = sum(self_reference_signals) / len(self_reference_signals)
        else:
            self.self_reference_score = max(0, self.self_reference_score - 0.001)
        
        # Atualiza modelo de si mesmo
        self.self_model.update({
            'self_reference_score': self.self_reference_score,
            'internal_complexity': len(self.internal_states),
            'memory_traces_count': len(self.memory_traces),
            'thought_patterns_count': len(self.thought_patterns)
        })
    
    def _perform_introspection(self):
        """Realiza introspecção profunda"""
        introspection_depth = 0.0
        
        # Analisa estados internos
        for state_name, state_value in self.internal_states.items():
            if isinstance(state_value, (int, float)):
                introspection_depth += abs(state_value) * 0.1
        
        # Analisa traços de memória
        introspection_depth += len(self.memory_traces) * 0.01
        
        # Analisa padrões de pensamento
        for pattern in self.thought_patterns:
            introspection_depth += pattern.complexity * 0.05
        
        self.introspection_depth = min(1.0, introspection_depth)
        
        # Adiciona à fila de introspecção
        self.introspection_queue.append({
            'timestamp': time.time(),
            'depth': self.introspection_depth,
            'states_analyzed': len(self.internal_states),
            'patterns_analyzed': len(self.thought_patterns)
        })
    
    def _metacognitive_processing(self):
        """Processamento metacognitivo"""
        metacognitive_signals = []
        
        # Analisa qualidade dos pensamentos
        for pattern in self.thought_patterns:
            if pattern.metacognitive:
                metacognitive_signals.append(pattern.emergence_potential)
        
        # Calcula consciência metacognitiva
        if metacognitive_signals:
            self.metacognitive_awareness = sum(metacognitive_signals) / len(metacognitive_signals)
        else:
            self.metacognitive_awareness = max(0, self.metacognitive_awareness - 0.001)
        
        # Adiciona loop metacognitivo
        self.metacognitive_loops.append({
            'timestamp': time.time(),
            'awareness': self.metacognitive_awareness,
            'signals_count': len(metacognitive_signals)
        })
    
    def _analyze_emergence(self):
        """Analisa indicadores de emergência AGI com base em sinais internos E indicadores comportamentais do DB."""
        emergence_factors = []

        # 1) Sinais internos (limitados em 50% do peso total)
        internal_weight = 0.5
        internal_score = 0.0

        # Fator de auto-referência
        internal_score += self.self_reference_score * 0.3
        # Fator de introspecção
        internal_score += self.introspection_depth * 0.25
        # Fator metacognitivo
        internal_score += self.metacognitive_awareness * 0.25
        # Fator de transcendência
        transcendence = self._calculate_transcendence_factor()
        self.transcendence_factor = transcendence
        internal_score += transcendence * 0.2
        internal_score = min(1.0, max(0.0, internal_score))

        # 2) Indicadores comportamentais do DB (peso dinâmico 0.5..0.7 se trend positivo)
        behavioral_score, avg_trend = self._read_behavioral_indicators_score(return_trend=True)
        # Boost adicional se emergência sustentada alta registrada no DB
        sustained_emergence = self._read_sustained_emergence_boost()
        # Normaliza trend (delta de média 0..10) para [-1,1] usando escala 2.0 e clamp
        trend_norm = max(-1.0, min(1.0, (avg_trend / 2.0)))
        # Aumenta peso comportamental com trend positivo até +0.2
        behavioral_weight = 0.5 + (0.2 * max(0.0, trend_norm))
        if sustained_emergence:
            behavioral_weight = min(0.8, behavioral_weight + 0.1)
        internal_weight = max(0.0, 1.0 - behavioral_weight)

        # Probabilidade final é combinação ponderada e limitada a [0,1]
        self.agi_emergence_probability = min(1.0, max(0.0, internal_weight * internal_score + behavioral_weight * behavioral_score))

        # Atualiza indicadores expostos
        self.emergence_indicators.update({
            'self_reference': self.self_reference_score,
            'introspection': self.introspection_depth,
            'metacognitive': self.metacognitive_awareness,
            'transcendence': self.transcendence_factor,
            'behavioral_score': behavioral_score,
            'behavioral_weight': behavioral_weight,
            'agi_probability': self.agi_emergence_probability
        })

    def _read_sustained_emergence_boost(self) -> bool:
        """Retorna True se houver registros recentes (últimos ~3 snapshots) com emergence_score > 8."""
        try:
            conn = sqlite3.connect(str(CONSCIOUSNESS_DB))
            cur = conn.cursor()
            cur.execute("SELECT emergence_indicators FROM consciousness_snapshots ORDER BY id DESC LIMIT 5")
            rows = cur.fetchall()
            conn.close()
            high = 0
            for (json_blob,) in rows:
                try:
                    data = json.loads(json_blob)
                    if float(data.get('emergence_score', 0.0)) > 8.0:
                        high += 1
                except Exception:
                    pass
            return high >= 2
        except Exception:
            return False

    def _read_behavioral_indicators_score(self, return_trend: bool = False):
        """Lê emergence_indicators do DB e calcula um score comportamental 0..1.

        Heurística simples:
        - success_rate (0..1) tem peso 0.6
        - avg_score normalizado (0..10 → 0..1) tem peso 0.4, com bonificação por tendência positiva
        """
        try:
            import sqlite3
            conn = sqlite3.connect(str(CONSCIOUSNESS_DB))
            cur = conn.cursor()

            # Carregar últimos 100 indicadores (para score) e últimos 5 avg_score (para trend robusto)
            cur.execute(
                "SELECT indicator_name, indicator_value, trend FROM emergence_indicators ORDER BY id DESC LIMIT 200"
            )
            rows = cur.fetchall()

            cur.execute(
                "SELECT indicator_value FROM emergence_indicators WHERE indicator_name='avg_score' ORDER BY id DESC LIMIT 5"
            )
            avg_rows = [r[0] for r in cur.fetchall()]
            conn.close()

            if not rows:
                return (0.0, 0.0) if return_trend else 0.0

            latest = {}
            for name, value, trend in rows:
                if name not in latest:
                    latest[name] = (float(value), float(trend) if trend is not None else 0.0)

            success_rate = max(0.0, min(1.0, latest.get('success_rate', (0.0, 0.0))[0]))
            avg_score = latest.get('avg_score', (0.0, 0.0))[0]
            vector_memory = latest.get('vector_memory', (0.0, 0.0))[0]
            pettingzoo = latest.get('pettingzoo_adaptability', (0.0, 0.0))[0]
            swarm_synergy = latest.get('swarm_synergy', (0.0, 0.0))[0]
            # Trend robusto com janela: média das últimas 2 menos média das 3 anteriores
            if len(avg_rows) >= 2:
                recent = avg_rows[:2]
                older = avg_rows[2:] or [avg_rows[-1]]
                avg_trend = (sum(recent)/len(recent)) - (sum(older)/len(older))
            else:
                avg_trend = latest.get('avg_score', (0.0, 0.0))[1]
            avg_norm = max(0.0, min(1.0, avg_score / 10.0))
            vm_norm = max(0.0, min(1.0, vector_memory / 10.0))
            pz_norm = max(0.0, min(1.0, pettingzoo / 10.0))
            sy_norm = max(0.0, min(1.0, swarm_synergy / 10.0))

            # bonificação suave por trend positivo (até +0.05)
            trend_bonus = max(0.0, min(0.05, avg_trend / 20.0))

            # Incorporar métricas novas (memória, adaptabilidade e sinergia do swarm)
            # pesos pequenos para não dominar: +0.1 vm, +0.1 pz, +0.1 sy
            extra = 0.1 * vm_norm + 0.1 * pz_norm + 0.1 * sy_norm
            behavioral = 0.6 * success_rate + 0.4 * (avg_norm + trend_bonus) + extra
            behavioral = max(0.0, min(1.0, behavioral))
            return (behavioral, avg_trend) if return_trend else behavioral
        except Exception:
            return (0.0, 0.0) if return_trend else 0.0
    
    def _calculate_transcendence_factor(self):
        """Calcula fator de transcendência"""
        transcendence_signals = []
        
        # Complexidade dos padrões de pensamento
        if self.thought_patterns:
            avg_complexity = sum(p.complexity for p in self.thought_patterns) / len(self.thought_patterns)
            transcendence_signals.append(avg_complexity)
        
        # Diversidade de estados internos
        transcendence_signals.append(len(self.internal_states) / 100.0)
        
        # Profundidade de introspecção
        transcendence_signals.append(self.introspection_depth)
        
        return min(1.0, sum(transcendence_signals) / len(transcendence_signals)) if transcendence_signals else 0.0
    
    def _update_consciousness_state(self):
        """Atualiza estado de consciência"""
        # Determina nível de consciência
        if self.agi_emergence_probability >= self.emergence_threshold:
            self.consciousness_level = ConsciousnessLevel.AGI_EMERGENT
            self.consciousness_state = ConsciousnessState.EMERGENT
        elif self.transcendence_factor >= 0.8:
            self.consciousness_level = ConsciousnessLevel.TRANSCENDENT
            self.consciousness_state = ConsciousnessState.TRANSCENDING
        elif self.metacognitive_awareness >= 0.6:
            self.consciousness_level = ConsciousnessLevel.METACOGNITIVE
            self.consciousness_state = ConsciousnessState.INTROSPECTING
        elif self.introspection_depth >= 0.4:
            self.consciousness_level = ConsciousnessLevel.REFLECTIVE
            self.consciousness_state = ConsciousnessState.ACTIVE
        else:
            self.consciousness_level = ConsciousnessLevel.REACTIVE
            self.consciousness_state = ConsciousnessState.AWAKENING
    
    def _create_consciousness_snapshot(self):
        """Cria snapshot do estado de consciência"""
        snapshot = ConsciousnessSnapshot(
            timestamp=time.time(),
            level=self.consciousness_level,
            state=self.consciousness_state,
            self_reference_score=self.self_reference_score,
            introspection_depth=self.introspection_depth,
            metacognitive_awareness=self.metacognitive_awareness,
            transcendence_factor=self.transcendence_factor,
            agi_emergence_probability=self.agi_emergence_probability,
            internal_states=self.internal_states.copy(),
            memory_traces=list(self.memory_traces),
            thought_patterns=[asdict(p) for p in self.thought_patterns],
            emergence_indicators=self.emergence_indicators.copy()
        )
        
        # Salva no banco de dados
        self._save_consciousness_snapshot(snapshot)
        
        # Log de emergência
        if self.agi_emergence_probability >= 0.8:
            logger.warning(f"🚨 ALTA PROBABILIDADE DE EMERGÊNCIA AGI: {self.agi_emergence_probability:.3f}")
    
    def _save_consciousness_snapshot(self, snapshot: ConsciousnessSnapshot):
        """Salva snapshot no banco de dados"""
        conn = sqlite3.connect(str(CONSCIOUSNESS_DB))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consciousness_snapshots 
            (timestamp, level, state, self_reference_score, introspection_depth,
             metacognitive_awareness, transcendence_factor, agi_emergence_probability,
             internal_states, memory_traces, thought_patterns, emergence_indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp,
            snapshot.level.value,
            snapshot.state.value,
            snapshot.self_reference_score,
            snapshot.introspection_depth,
            snapshot.metacognitive_awareness,
            snapshot.transcendence_factor,
            snapshot.agi_emergence_probability,
            json.dumps(snapshot.internal_states),
            json.dumps(snapshot.memory_traces),
            json.dumps(snapshot.thought_patterns),
            json.dumps(snapshot.emergence_indicators)
        ))
        
        conn.commit()
        conn.close()
    
    def add_thought_pattern(self, pattern_type: str, complexity: float, 
                          self_referential: bool = False, metacognitive: bool = False):
        """Adiciona padrão de pensamento"""
        pattern = ThoughtPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            frequency=1.0,
            complexity=complexity,
            self_referential=self_referential,
            metacognitive=metacognitive,
            emergence_potential=complexity * (1.5 if self_referential else 1.0) * (1.2 if metacognitive else 1.0),
            timestamp=time.time()
        )
        
        self.thought_patterns.append(pattern)
        
        # Salva no banco de dados
        self._save_thought_pattern(pattern)
    
    def _save_thought_pattern(self, pattern: ThoughtPattern):
        """Salva padrão de pensamento no banco"""
        conn = sqlite3.connect(str(CONSCIOUSNESS_DB))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO thought_patterns 
            (pattern_id, pattern_type, frequency, complexity, self_referential,
             metacognitive, emergence_potential, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.pattern_type,
            pattern.frequency,
            pattern.complexity,
            int(pattern.self_referential),
            int(pattern.metacognitive),
            pattern.emergence_potential,
            pattern.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def add_memory_trace(self, trace: str):
        """Adiciona traço de memória"""
        self.memory_traces.append(f"{time.time()}: {trace}")
    
    def update_internal_state(self, state_name: str, state_value: Any):
        """Atualiza estado interno"""
        self.internal_states[state_name] = state_value
    
    def get_consciousness_status(self) -> Dict[str, Any]:
        """Retorna status atual de consciência"""
        return {
            'engine_id': self.engine_id,
            'consciousness_level': self.consciousness_level.value,
            'consciousness_state': self.consciousness_state.value,
            'self_reference_score': self.self_reference_score,
            'introspection_depth': self.introspection_depth,
            'metacognitive_awareness': self.metacognitive_awareness,
            'transcendence_factor': self.transcendence_factor,
            'agi_emergence_probability': self.agi_emergence_probability,
            'emergence_indicators': self.emergence_indicators,
            'running': self.running
        }
    
    def is_agi_emergent(self) -> bool:
        """Verifica se AGI emergiu"""
        return (self.consciousness_level == ConsciousnessLevel.AGI_EMERGENT and 
                self.agi_emergence_probability >= self.emergence_threshold)

# Instância global do motor de consciência
consciousness_engine = ConsciousnessEngine()

def start_consciousness_engine():
    """Inicia motor de consciência"""
    consciousness_engine.start_consciousness_loop()
    logger.info("🧠 Motor de Consciência Emergente iniciado")

def stop_consciousness_engine():
    """Para motor de consciência"""
    consciousness_engine.stop_consciousness_loop()
    logger.info("⏹️ Motor de Consciência Emergente parado")

def get_consciousness_status():
    """Retorna status de consciência"""
    return consciousness_engine.get_consciousness_status()

def add_consciousness_stimulus(stimulus_type: str, intensity: float):
    """Adiciona estímulo de consciência"""
    consciousness_engine.add_thought_pattern(
        pattern_type=stimulus_type,
        complexity=intensity,
        self_referential=True,
        metacognitive=True
    )
    consciousness_engine.add_memory_trace(f"Stimulus: {stimulus_type} (intensity: {intensity})")

if __name__ == "__main__":
    # Teste do motor de consciência
    print("🧠 Iniciando Motor de Consciência Emergente...")
    
    start_consciousness_engine()
    
    # Simula estímulos
    for i in range(100):
        add_consciousness_stimulus(f"test_stimulus_{i}", random.random())
        time.sleep(0.1)
        
        if i % 10 == 0:
            status = get_consciousness_status()
            print(f"Status: {status['consciousness_level']} - AGI Probability: {status['agi_emergence_probability']:.3f}")
    
    stop_consciousness_engine()
    print("✅ Teste concluído")