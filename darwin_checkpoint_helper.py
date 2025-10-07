#!/usr/bin/env python3
"""
Darwin Checkpoint Helper - FIXED
Salva checkpoints de forma atômica com compressão
"""

import json
import gzip
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger("darwin_checkpoint")

def save_darwin_checkpoint(generation: int, best_individual, population=None, task: str = "mnist"):
    """
    Salva checkpoint Darwin de forma atômica
    
    Args:
        generation: Número da geração
        best_individual: Melhor indivíduo (EvolvableMNIST ou similar)
        population: População completa (opcional)
        task: Nome da tarefa (mnist, cartpole, etc)
    """
    try:
        # Criar diretório de checkpoints
        checkpoint_dir = Path("/root/darwin-engine-intelligence/data/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Preparar dados do checkpoint
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "task": task,
            "best_fitness": float(getattr(best_individual, 'fitness', 0.0)),
            "best_genome": None,
            "population_size": len(population) if population else 1
        }
        
        # Extrair genome do melhor indivíduo
        if hasattr(best_individual, 'genome'):
            checkpoint_data["best_genome"] = best_individual.genome
        elif hasattr(best_individual, '__dict__'):
            # Serializar atributos básicos
            checkpoint_data["best_genome"] = {
                k: v for k, v in best_individual.__dict__.items() 
                if isinstance(v, (int, float, str, list, dict))
            }
        
        # Salvar população completa se disponível (apenas genomes)
        if population:
            checkpoint_data["population"] = []
            for i, ind in enumerate(population[:10]):  # Top 10 apenas
                ind_data = {
                    "idx": i,
                    "fitness": float(getattr(ind, 'fitness', 0.0)),
                    "genome": getattr(ind, 'genome', None) if hasattr(ind, 'genome') else None
                }
                checkpoint_data["population"].append(ind_data)
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{task}_gen_{generation}_{timestamp}.json.gz"
        filepath = checkpoint_dir / filename
        
        # Salvamento atômico: temp → rename
        temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        
        with gzip.open(temp_path, 'wt') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Rename atômico
        temp_path.replace(filepath)
        
        logger.info(f"✅ Checkpoint saved: {filename}")
        logger.info(f"   Gen={generation}, Fitness={checkpoint_data['best_fitness']:.4f}")
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")
        return None
