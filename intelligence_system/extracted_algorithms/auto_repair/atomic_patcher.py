"""
ATOMIC PATCHER - Aplicação segura de patches
Aplica mudanças de código de forma atômica com backup e rollback
"""
import os
import shutil
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class AtomicPatcher:
    """Aplica patches atomicamente com backup e validação"""
    
    def __init__(self, backup_dir: str = "/root/intelligence_system/auto_repair_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.applied_patches = []
        self.rate_limit = 3  # patches por hora
        self.last_patch_times = []
        
    def can_apply_patch(self) -> bool:
        """Verifica rate limit"""
        now = time.time()
        hour_ago = now - 3600
        
        # Remove patches antigos
        self.last_patch_times = [t for t in self.last_patch_times if t > hour_ago]
        
        return len(self.last_patch_times) < self.rate_limit
    
    def apply_patch(self, target_file: str, 
                    new_content: str,
                    patch_metadata: Dict,
                    dry_run: bool = False) -> Dict:
        """
        Aplica patch atomicamente
        
        Args:
            target_file: arquivo alvo
            new_content: novo conteúdo
            patch_metadata: metadados do patch
            dry_run: se True, apenas simula
            
        Returns:
            Resultado da operação
        """
        result = {
            'success': False,
            'target': target_file,
            'backup_path': None,
            'applied': False,
            'validated': False,
            'error': None,
            'dry_run': dry_run,
            'timestamp': time.time()
        }
        
        target_path = Path(target_file)
        
        # Verificações prévias
        if not target_path.exists():
            result['error'] = f"Target file does not exist: {target_file}"
            return result
        
        if not self.can_apply_patch():
            result['error'] = "Rate limit exceeded (3 patches/hour)"
            return result
        
        try:
            # Lê conteúdo original
            with open(target_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Hash do original
            original_hash = hashlib.sha256(original_content.encode()).hexdigest()
            
            # Cria backup
            backup_path = self._create_backup(target_path, original_content)
            result['backup_path'] = str(backup_path)
            
            if dry_run:
                result['success'] = True
                result['message'] = "Dry-run: patch validated but not applied"
                return result
            
            # Aplica patch atomicamente
            temp_file = target_path.with_suffix('.tmp')
            
            # Escreve novo conteúdo no temp
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Replace atômico
            os.replace(temp_file, target_path)
            
            result['applied'] = True
            result['success'] = True
            self.last_patch_times.append(time.time())
            
            # Registra patch aplicado
            patch_record = {
                'target': target_file,
                'backup': str(backup_path),
                'original_hash': original_hash,
                'new_hash': hashlib.sha256(new_content.encode()).hexdigest(),
                'metadata': patch_metadata,
                'timestamp': result['timestamp']
            }
            self.applied_patches.append(patch_record)
            
            # Salva registro
            self._save_patch_record(patch_record)
            
            logger.info(f"✅ Patch applied: {target_file}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Patch failed: {e}")
            
            # Tenta rollback se aplicou parcialmente
            if result['applied'] and result['backup_path']:
                self._rollback(target_file, result['backup_path'])
        
        return result
    
    def _create_backup(self, target_path: Path, content: str) -> Path:
        """Cria backup do arquivo"""
        timestamp = int(time.time() * 1000)
        relative = target_path.relative_to('/root/intelligence_system')
        backup_path = self.backup_dir / f"{timestamp}_{relative.as_posix().replace('/', '_')}"
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return backup_path
    
    def _rollback(self, target_file: str, backup_path: str) -> bool:
        """Reverte arquivo para backup"""
        try:
            shutil.copy2(backup_path, target_file)
            logger.info(f"🔄 Rollback successful: {target_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def _save_patch_record(self, record: Dict):
        """Salva registro de patch"""
        records_file = self.backup_dir / 'patch_records.jsonl'
        
        with open(records_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    
    def validate_patch(self, target_file: str, 
                       validation_fn: Optional[callable] = None) -> bool:
        """
        Valida patch aplicado
        
        Args:
            target_file: arquivo modificado
            validation_fn: função customizada de validação
            
        Returns:
            True se válido
        """
        try:
            # Validação básica: arquivo existe e é válido Python
            target_path = Path(target_file)
            
            if not target_path.exists():
                return False
            
            # Tenta compilar (se for Python)
            if target_file.endswith('.py'):
                with open(target_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), target_file, 'exec')
            
            # Validação customizada
            if validation_fn:
                return validation_fn(target_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def get_patch_history(self) -> List[Dict]:
        """Retorna histórico de patches"""
        return self.applied_patches.copy()


class SmartPatcher(AtomicPatcher):
    """Patcher com estratégias inteligentes"""
    
    def apply_incremental_patch(self, target_file: str,
                                 search_pattern: str,
                                 replacement: str,
                                 patch_metadata: Dict,
                                 dry_run: bool = False) -> Dict:
        """
        Aplica patch incremental (busca e substitui)
        
        Mais seguro que substituir arquivo inteiro
        """
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if search_pattern not in content:
                return {
                    'success': False,
                    'error': f"Pattern not found in {target_file}",
                    'dry_run': dry_run
                }
            
            # Substitui
            new_content = content.replace(search_pattern, replacement, 1)
            
            # Aplica usando método atômico
            return self.apply_patch(target_file, new_content, patch_metadata, dry_run)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dry_run': dry_run
            }
    
    def apply_function_replacement(self, target_file: str,
                                   function_name: str,
                                   new_function_code: str,
                                   patch_metadata: Dict,
                                   dry_run: bool = False) -> Dict:
        """
        Substitui apenas uma função específica
        
        Mais cirúrgico e seguro
        """
        import ast
        import astor
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Encontra função
            function_found = False
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        # Parse nova função
                        new_func = ast.parse(new_function_code).body[0]
                        # Substitui no AST
                        tree.body[i] = new_func
                        function_found = True
                        break
            
            if not function_found:
                return {
                    'success': False,
                    'error': f"Function '{function_name}' not found in {target_file}",
                    'dry_run': dry_run
                }
            
            # Regenera código
            new_content = astor.to_source(tree)
            
            # Aplica
            return self.apply_patch(target_file, new_content, patch_metadata, dry_run)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dry_run': dry_run
            }


if __name__ == "__main__":
    # Teste
    patcher = AtomicPatcher()
    print(f"Can apply patch: {patcher.can_apply_patch()}")
    print(f"Backup dir: {patcher.backup_dir}")
