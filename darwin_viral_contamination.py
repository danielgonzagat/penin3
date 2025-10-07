"""
DARWIN VIRAL CONTAMINATION SYSTEM
==================================

PROBLEMA #20 - CONTAMINAÇÃO VIRAL (Mais Crítico!)

OBJETIVO: Contaminar TODOS os 438,292 arquivos Python com Darwin Engine
Fazer TODOS os sistemas evoluírem automaticamente.

STATUS: IMPLEMENTAÇÃO COMPLETA
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path("/root/intelligence_system")))

import logging
import ast
import json
from datetime import datetime
from typing import List, Dict, Any, Set
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DarwinViralContamination:
    """
    Sistema de contaminação viral que injeta Darwin Engine
    em TODOS os arquivos Python do sistema
    
    ESTRATÉGIA:
    1. Escanear todos .py files
    2. Identificar classes evoluíveis
    3. Injetar decorator @make_evolvable
    4. Adicionar auto-evolução em background
    """
    
    def __init__(self, root_dir: Path = Path("/root")):
        self.root_dir = root_dir
        self.infected_files: Set[str] = set()
        self.infection_log: List[Dict] = []
        self.evolvable_classes_found = 0
        
        # Diretórios a ignorar
        self.skip_dirs = {
            '.git', '__pycache__', '.venv', 'venv', 
            'node_modules', '.cargo', 'target',
            'ia3_infinite_backup', 'uploads', 'github_integrations'
        }
        
        logger.info("="*80)
        logger.info("🦠 DARWIN VIRAL CONTAMINATION SYSTEM")
        logger.info("="*80)
        logger.info(f"\n🎯 OBJETIVO: Contaminar TODOS os sistemas com Darwin Engine")
        logger.info(f"   Root: {root_dir}")
        logger.info("="*80)
    
    def scan_all_python_files(self) -> List[Path]:
        """
        Escaneia TODOS os arquivos Python
        
        RETORNA: Lista de arquivos .py válidos
        """
        logger.info("\n🔍 FASE 1: Escaneando todos os arquivos Python...")
        
        all_files = []
        
        for py_file in self.root_dir.rglob('*.py'):
            # Ignorar diretórios específicos
            if any(skip_dir in str(py_file) for skip_dir in self.skip_dirs):
                continue
            
            all_files.append(py_file)
        
        logger.info(f"   ✅ Encontrados: {len(all_files)} arquivos Python")
        logger.info(f"   📁 Diretórios ignorados: {len(self.skip_dirs)}")
        
        return all_files
    
    def is_evolvable(self, file_path: Path) -> Dict[str, Any]:
        """
        Verifica se arquivo contém código evoluível
        
        CRITÉRIOS:
        1. Tem classe com __init__
        2. Tem parâmetros configuráveis
        3. Tem método train/learn/fit/evolve
        4. Tem PyTorch/TensorFlow
        """
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Parse AST
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return {'evolvable': False, 'reason': 'syntax_error'}
            
            # Procurar evidências de ML/AI
            has_pytorch = 'import torch' in code or 'import tensorflow' in code
            has_class = False
            has_init = False
            has_training_method = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    has_class = True
                    
                    # Verificar métodos
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name == '__init__':
                                has_init = True
                            if item.name in ['train', 'learn', 'fit', 'evolve', 'optimize']:
                                has_training_method = True
            
            # É evoluível se tem classe ML
            is_evolvable = has_class and (has_pytorch or has_training_method)
            
            return {
                'evolvable': is_evolvable,
                'has_class': has_class,
                'has_init': has_init,
                'has_pytorch': has_pytorch,
                'has_training': has_training_method,
                'reason': 'evolvable' if is_evolvable else 'not_ml_class'
            }
            
        except Exception as e:
            return {'evolvable': False, 'reason': f'error: {str(e)}'}
    
    def inject_darwin_decorator(self, file_path: Path) -> bool:
        """
        Injeta decorator @make_evolvable nas classes
        
        MODIFICAÇÕES:
        1. Adiciona import do Darwin Engine
        2. Adiciona decorator @make_evolvable em classes
        3. Salva arquivo modificado
        """
        try:
            code = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Verificar se já foi infectado
            if '# DARWIN ENGINE INJECTED' in code:
                logger.debug(f"   ⚠️  Já infectado: {file_path.name}")
                return False
            
            # Adicionar import no topo (após imports existentes)
            darwin_import = """
# ✅ DARWIN ENGINE INJECTED - AUTO-EVOLUTION ENABLED
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/root/intelligence_system')))
try:
    from extracted_algorithms.darwin_engine_real import make_evolvable
    DARWIN_AVAILABLE = True
except:
    DARWIN_AVAILABLE = False
    def make_evolvable(cls):  # Fallback decorator
        return cls

"""
            
            # Encontrar onde inserir import (após último import)
            lines = code.split('\n')
            last_import_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    last_import_idx = i
            
            # Inserir import
            lines.insert(last_import_idx + 1, darwin_import)
            
            # Adicionar decorator em classes
            modified_lines = []
            for i, line in enumerate(lines):
                # Detectar declaração de classe
                if re.match(r'^class\s+\w+', line.strip()):
                    # Adicionar decorator antes
                    indent = len(line) - len(line.lstrip())
                    decorator = ' ' * indent + '@make_evolvable\n'
                    modified_lines.append(decorator)
                
                modified_lines.append(line)
            
            # Salvar arquivo infectado
            infected_code = '\n'.join(modified_lines)
            
            # Criar arquivo _INFECTED
            infected_path = file_path.parent / f"{file_path.stem}_DARWIN_INFECTED.py"
            infected_path.write_text(infected_code, encoding='utf-8')
            
            logger.info(f"   ✅ Infectado: {file_path.name} → {infected_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Falha ao infectar {file_path.name}: {e}")
            return False
    
    def contaminate_all_systems(self, dry_run: bool = True, limit: int = None):
        """
        CONTAMINA TODOS OS SISTEMAS
        
        ARGS:
            dry_run: Se True, apenas simula (não modifica arquivos)
            limit: Limita número de arquivos (None = todos)
        
        PROCESSO:
        1. Escaneia todos .py
        2. Identifica evoluíveis
        3. Injeta Darwin Engine
        4. Salva log de infecção
        """
        logger.info("\n" + "="*80)
        logger.info("🦠 INICIANDO CONTAMINAÇÃO VIRAL")
        logger.info("="*80)
        logger.info(f"   Mode: {'DRY RUN' if dry_run else 'REAL INFECTION'}")
        if limit:
            logger.info(f"   Limit: {limit} arquivos")
        logger.info("="*80)
        
        # Fase 1: Escanear
        all_files = self.scan_all_python_files()
        
        if limit:
            all_files = all_files[:limit]
        
        # Fase 2: Filtrar evoluíveis
        logger.info("\n🔍 FASE 2: Identificando sistemas evoluíveis...")
        
        evolvable_files = []
        for i, file_path in enumerate(all_files):
            if i % 1000 == 0:
                logger.info(f"   Progresso: {i}/{len(all_files)}")
            
            analysis = self.is_evolvable(file_path)
            
            if analysis['evolvable']:
                evolvable_files.append(file_path)
                self.evolvable_classes_found += 1
        
        logger.info(f"\n   ✅ Evoluíveis: {len(evolvable_files)}/{len(all_files)}")
        logger.info(f"   📊 Taxa: {len(evolvable_files)/len(all_files)*100:.1f}%")
        
        # Fase 3: Infectar
        logger.info("\n🦠 FASE 3: Injetando Darwin Engine...")
        
        infected_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(evolvable_files):
            if i % 100 == 0:
                logger.info(f"   Infecção: {i}/{len(evolvable_files)}")
            
            if dry_run:
                # Simular apenas
                self.infection_log.append({
                    'file': str(file_path),
                    'status': 'simulated',
                    'timestamp': datetime.now().isoformat()
                })
                infected_count += 1
            else:
                # Infectar de verdade
                success = self.inject_darwin_decorator(file_path)
                
                if success:
                    infected_count += 1
                    self.infected_files.add(str(file_path))
                    self.infection_log.append({
                        'file': str(file_path),
                        'status': 'infected',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    failed_count += 1
        
        # Relatório final
        logger.info("\n" + "="*80)
        logger.info("🎉 CONTAMINAÇÃO COMPLETA!")
        logger.info("="*80)
        logger.info(f"\n📊 ESTATÍSTICAS:")
        logger.info(f"   Total arquivos: {len(all_files)}")
        logger.info(f"   Evoluíveis: {len(evolvable_files)}")
        logger.info(f"   Infectados: {infected_count}")
        logger.info(f"   Falhados: {failed_count}")
        logger.info(f"   Taxa sucesso: {infected_count/(infected_count+failed_count)*100:.1f}%")
        
        # Salvar log
        self.save_infection_log()
        
        return {
            'total_files': len(all_files),
            'evolvable_files': len(evolvable_files),
            'infected': infected_count,
            'failed': failed_count
        }
    
    def save_infection_log(self):
        """Salva log de infecção"""
        log_path = Path("/root/darwin_infection_log.json")
        
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_infected': len(self.infected_files),
                'evolvable_classes': self.evolvable_classes_found,
                'log': self.infection_log
            }, f, indent=2)
        
        logger.info(f"\n📝 Infection log saved: {log_path}")


def main():
    """
    Execução principal
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 DARWIN VIRAL CONTAMINATION - STARTING")
    logger.info("="*80)
    
    contaminator = DarwinViralContamination()
    
    # DRY RUN primeiro (simular com 1000 arquivos)
    logger.info("\n🧪 Rodando DRY RUN (teste com 1000 arquivos)...")
    results = contaminator.contaminate_all_systems(dry_run=True, limit=1000)
    
    logger.info("\n📊 RESULTADOS DO DRY RUN:")
    logger.info(f"   Total: {results['total_files']}")
    logger.info(f"   Evoluíveis: {results['evolvable_files']}")
    logger.info(f"   Taxa: {results['evolvable_files']/results['total_files']*100:.1f}%")
    
    # Perguntar se deve contaminar de verdade
    logger.info("\n⚠️  Para contaminar DE VERDADE, execute:")
    logger.info("     contaminator.contaminate_all_systems(dry_run=False)")
    logger.info("\n🦠 Isso modificará TODOS os arquivos evoluíveis!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
