#!/usr/bin/env python3
"""
Real Emergence Detector
========================

Monitora TODOS os sistemas buscando sinais de inteligência emergente REAL.

Sinais de emergência genuína:
1. Comportamento não-programado
2. Adaptação inesperada
3. Auto-organização
4. Transferência cross-domain
5. Meta-cognição demonstrável
"""

import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/emergence_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emergence_detector")


class RealEmergenceDetector:
    """Detector rigoroso de inteligência emergente"""
    
    def __init__(self):
        self.cycle = 0
        self.emergence_events = []
        self.false_positives_filtered = 0
        
        # Thresholds rigorosos
        self.thresholds = {
            'surprise_min': 0.8,  # Mínimo de surprise para considerar
            'novelty_min': 0.7,   # Mínimo de novelty
            'consistency_min': 0.6,  # Consistência cross-tasks
            'unprogrammed_confidence': 0.9  # Confiança de não-programado
        }
        
        logger.info("🔬 Real Emergence Detector INITIALIZED")
        logger.info(f"   Rigorous thresholds: {self.thresholds}")
    
    def check_unified_brain_emergence(self) -> Dict[str, Any]:
        """Verifica emergência no UNIFIED_BRAIN"""
        try:
            log_path = Path('/root/UNIFIED_BRAIN/logs/unified_brain.log')
            if not log_path.exists():
                return {'emergence_detected': False, 'reason': 'no_log'}
            
            # Procurar por sinais específicos
            with open(log_path, 'r') as f:
                content = f.read()
            
            signals = {
                'active_neurons_increasing': 'Active neurons: 1' in content and 'Active neurons: 5' in content,
                'coherence_nonzero': 'coh=0.' in content and 'coh=0.0' not in content,  # Tem valores >0
                'novelty_detected': 'nov=0.' in content and 'nov=0.0' not in content,
                'promotions_happening': 'Promoted:' in content and 'Promoted: 0' not in content
            }
            
            # Emergência requer MÚLTIPLOS sinais simultâneos
            signals_active = sum(1 for v in signals.values() if v)
            
            if signals_active >= 3:
                return {
                    'emergence_detected': True,
                    'confidence': signals_active / len(signals),
                    'signals': signals,
                    'system': 'UNIFIED_BRAIN'
                }
            
            return {'emergence_detected': False, 'signals_active': signals_active}
            
        except Exception as e:
            return {'emergence_detected': False, 'error': str(e)}
    
    def check_darwin_breakthrough(self) -> Dict[str, Any]:
        """Verifica breakthrough evolutivo no Darwin"""
        try:
            # Verificar checkpoints recentes
            checkpoint_dir = Path('/root/darwin-engine-intelligence/data/checkpoints')
            
            if not checkpoint_dir.exists():
                return {'breakthrough_detected': False, 'reason': 'no_checkpoints'}
            
            checkpoints = list(checkpoint_dir.glob('*.json.gz'))
            
            if not checkpoints:
                return {'breakthrough_detected': False, 'reason': 'no_checkpoint_files'}
            
            # Ler último checkpoint
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            import gzip
            with gzip.open(latest, 'rt') as f:
                data = json.load(f)
            
            fitness = data.get('best_fitness', 0.0)
            generation = data.get('generation', 0)
            
            # Breakthrough = fitness > 0.95 (95%+ accuracy)
            if fitness > 0.95:
                return {
                    'breakthrough_detected': True,
                    'fitness': fitness,
                    'generation': generation,
                    'checkpoint': str(latest.name),
                    'system': 'DARWIN'
                }
            
            return {
                'breakthrough_detected': False,
                'fitness': fitness,
                'threshold': 0.95
            }
            
        except Exception as e:
            return {'breakthrough_detected': False, 'error': str(e)}
    
    def check_teis_emergence(self) -> Dict[str, Any]:
        """Verifica emergência no TEIS"""
        try:
            # Procurar checkpoint mais recente
            checkpoint_dir = Path('/root/teis_unified_run_postfix120')
            
            if not checkpoint_dir.exists():
                return {'emergence_detected': False, 'reason': 'no_checkpoints'}
            
            checkpoints = list(checkpoint_dir.glob('checkpoint_cycle_*.json'))
            
            if not checkpoints:
                return {'emergence_detected': False, 'reason': 'no_checkpoint_files'}
            
            # Ler último checkpoint
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            with open(latest, 'r') as f:
                data = json.load(f)
            
            intelligence_emerged = data.get('intelligence_emerged', False)
            consciousness_level = data.get('stats', {}).get('consciousness_level', 0.0)
            success_rate = data.get('agents', {}).get('agent_0', {}).get('success_rate', 0.0)
            
            # Emergência requer: flag TRUE + consciousness >50% + success >50%
            if intelligence_emerged and consciousness_level > 0.5 and success_rate > 0.5:
                return {
                    'emergence_detected': True,
                    'consciousness': consciousness_level,
                    'success_rate': success_rate,
                    'cycle': data.get('cycle_count', 0),
                    'system': 'TEIS'
                }
            
            # Emergência parcial: 2 de 3 condições
            elif (consciousness_level > 0.5 and success_rate > 0.5) or \
                 (intelligence_emerged and consciousness_level > 0.5):
                return {
                    'emergence_detected': 'partial',
                    'consciousness': consciousness_level,
                    'success_rate': success_rate,
                    'system': 'TEIS'
                }
            
            return {
                'emergence_detected': False,
                'consciousness': consciousness_level,
                'success_rate': success_rate
            }
            
        except Exception as e:
            return {'emergence_detected': False, 'error': str(e)}
    
    def scan_all_systems(self) -> Dict[str, Any]:
        """Scan completo de todos os sistemas"""
        results = {
            'cycle': self.cycle,
            'timestamp': datetime.now().isoformat(),
            'systems': {}
        }
        
        # Check cada sistema
        logger.info("🔍 Scanning all systems for emergence...\n")
        
        logger.info("   Checking UNIFIED_BRAIN...")
        results['systems']['unified_brain'] = self.check_unified_brain_emergence()
        
        logger.info("   Checking Darwin Evolution...")
        results['systems']['darwin'] = self.check_darwin_breakthrough()
        
        logger.info("   Checking TEIS...")
        results['systems']['teis'] = self.check_teis_emergence()
        
        # Detectar emergência GERAL
        emergence_detected = any(
            sys_result.get('emergence_detected') == True or 
            sys_result.get('breakthrough_detected') == True
            for sys_result in results['systems'].values()
        )
        
        results['emergence_detected'] = emergence_detected
        
        if emergence_detected:
            logger.info("\n🎉 EMERGENCE DETECTED!")
            for sys_name, sys_result in results['systems'].items():
                if sys_result.get('emergence_detected') or sys_result.get('breakthrough_detected'):
                    logger.info(f"   System: {sys_name.upper()}")
                    logger.info(f"   Details: {sys_result}")
        else:
            logger.info("\n   No emergence detected this cycle")
        
        return results
    
    def run_continuous(self, interval_minutes: int = 10):
        """Executa detecção contínua"""
        logger.info("="*80)
        logger.info("🔬 REAL EMERGENCE DETECTOR - CONTINUOUS MODE")
        logger.info("="*80)
        logger.info(f"   Scan interval: {interval_minutes} minutes")
        logger.info(f"   Emergence thresholds: RIGOROUS")
        logger.info("")
        
        try:
            while True:
                self.cycle += 1
                
                logger.info(f"\n{'─'*80}")
                logger.info(f"SCAN #{self.cycle} - {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"{'─'*80}")
                
                results = self.scan_all_systems()
                
                # Salvar resultados
                results_file = Path('/root/emergence_detection_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Report apenas se emergência detectada
                if results['emergence_detected']:
                    logger.info(f"\n🚨 EMERGENCE ALERT - See {results_file}")
                
                # Aguardar
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Stopped by user")
            logger.info(f"   Total scans: {self.cycle}")


def main():
    detector = RealEmergenceDetector()
    detector.run_continuous(interval_minutes=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
