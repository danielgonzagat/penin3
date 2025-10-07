#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Organizador de Sistema de Arquivos
============================================
Limpa duplicatas, organiza arquivos e corrige paths inconsistentes.
"""

from __future__ import annotations
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
import json
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
BACKUP_PATH = Path("/root/.penin_omega_backup")

# =============================================================================
# ORGANIZADOR DE ARQUIVOS
# =============================================================================

class FileOrganizer:
    """Organiza e limpa sistema de arquivos."""
    
    async def __init__(self):
        self.logger = logging.getLogger("FileOrganizer")
        self.duplicates_found = {}
        self.files_moved = []
        self.files_deleted = []
        self.errors = []
        
    async def scan_duplicates(self, root_path: Path = Path("/root")) -> Dict[str, List[Path]]:
        """Escaneia arquivos duplicados baseado em hash MD5."""
        file_hashes = {}
        duplicates = {}
        
        # Padrões de arquivos PENIN-Ω
        penin_patterns = [
            "penin_omega_*.py",
            "*penin*.py", 
            "*.log",
            "*.json",
            "*.db",
            "*.sqlite*"
        ]
        
        self.logger.info("🔍 Escaneando duplicatas...")
        
        for pattern in penin_patterns:
            for file_path in root_path.rglob(pattern):
                if file_path.is_file() and file_path.stat().st_size > 0:
                    try:
                        # Calcula hash do arquivo
                        file_hash = self._calculate_file_hash(file_path)
                        
                        if file_hash in file_hashes:
                            # Duplicata encontrada
                            if file_hash not in duplicates:
                                duplicates[file_hash] = [file_hashes[file_hash]]
                            duplicates[file_hash].append(file_path)
                        else:
                            file_hashes[file_hash] = file_path
                            
                    except Exception as e:
                        self.errors.append(f"Erro ao processar {file_path}: {e}")
        
        self.duplicates_found = duplicates
        self.logger.info(f"✅ Escaneamento concluído: {len(duplicates)} grupos de duplicatas")
        return await duplicates
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash MD5 de um arquivo."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return await hash_md5.hexdigest()
    
    async def clean_duplicates(self, keep_newest: bool = True) -> int:
        """Remove arquivos duplicados."""
        if not self.duplicates_found:
            self.scan_duplicates()
        
        removed_count = 0
        
        for file_hash, file_list in self.duplicates_found.items():
            if len(file_list) <= 1:
                continue
                
            # Ordena por data de modificação
            sorted_files = sorted(file_list, key=lambda x: x.stat().st_mtime, reverse=keep_newest)
            
            # Mantém o primeiro (mais novo se keep_newest=True)
            keep_file = sorted_files[0]
            remove_files = sorted_files[1:]
            
            self.logger.info(f"📁 Mantendo: {keep_file}")
            
            for remove_file in remove_files:
                try:
                    # Backup antes de remover
                    self._backup_file(remove_file)
                    
                    # Remove arquivo
                    remove_file.unlink()
                    self.files_deleted.append(str(remove_file))
                    removed_count += 1
                    
                    self.logger.info(f"🗑️  Removido: {remove_file}")
                    
                except Exception as e:
                    self.errors.append(f"Erro ao remover {remove_file}: {e}")
        
        self.logger.info(f"✅ Limpeza concluída: {removed_count} arquivos removidos")
        return await removed_count
    
    async def _backup_file(self, file_path: Path):
        """Faz backup de arquivo antes de remover."""
        try:
            BACKUP_PATH.mkdir(parents=True, exist_ok=True)
            backup_file = BACKUP_PATH / file_path.name
            
            # Se backup já existe, adiciona timestamp
            if backup_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = BACKUP_PATH / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            shutil.copy2(file_path, backup_file)
            
        except Exception as e:
            self.logger.warning(f"Falha no backup de {file_path}: {e}")
    
    async def organize_penin_files(self) -> bool:
        """Organiza arquivos PENIN-Ω em estrutura padronizada."""
        try:
            # Estrutura de diretórios padrão
            standard_dirs = {
                "modules": PENIN_OMEGA_ROOT / "modules",
                "logs": PENIN_OMEGA_ROOT / "logs", 
                "config": PENIN_OMEGA_ROOT / "config",
                "cache": PENIN_OMEGA_ROOT / "cache",
                "artifacts": PENIN_OMEGA_ROOT / "artifacts",
                "knowledge": PENIN_OMEGA_ROOT / "knowledge",
                "worm": PENIN_OMEGA_ROOT / "worm",
                "tests": PENIN_OMEGA_ROOT / "tests",
                "backup": PENIN_OMEGA_ROOT / "backup"
            }
            
            # Cria diretórios
            for dir_name, dir_path in standard_dirs.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Diretório criado/verificado: {dir_path}")
            
            # Move arquivos para locais apropriados
            root_path = Path("/root")
            
            # Módulos Python
            for py_file in root_path.glob("penin_omega_*.py"):
                if py_file.is_file():
                    dest = standard_dirs["modules"] / py_file.name
                    if not dest.exists():
                        shutil.move(str(py_file), str(dest))
                        self.files_moved.append(f"{py_file} -> {dest}")
                        self.logger.info(f"📦 Movido: {py_file.name}")
            
            # Arquivos de log
            for log_file in root_path.glob("*.log"):
                if log_file.is_file():
                    dest = standard_dirs["logs"] / log_file.name
                    if not dest.exists():
                        shutil.move(str(log_file), str(dest))
                        self.files_moved.append(f"{log_file} -> {dest}")
            
            # Arquivos de configuração JSON
            for json_file in root_path.glob("*config*.json"):
                if json_file.is_file():
                    dest = standard_dirs["config"] / json_file.name
                    if not dest.exists():
                        shutil.move(str(json_file), str(dest))
                        self.files_moved.append(f"{json_file} -> {dest}")
            
            self.logger.info(f"✅ Organização concluída: {len(self.files_moved)} arquivos movidos")
            return await True
            
        except Exception as e:
            self.logger.error(f"Erro na organização: {e}")
            return await False
    
    async def fix_paths_in_configs(self) -> bool:
        """Corrige paths inconsistentes em arquivos de configuração."""
        try:
            config_files = list(PENIN_OMEGA_ROOT.rglob("*.json"))
            fixed_count = 0
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Corrige paths conhecidos
                    changes_made = False
                    
                    if isinstance(config_data, dict):
                        changes_made = self._fix_paths_recursive(config_data)
                    
                    if changes_made:
                        # Backup do arquivo original
                        backup_file = config_file.with_suffix('.json.bak')
                        shutil.copy2(config_file, backup_file)
                        
                        # Salva arquivo corrigido
                        with open(config_file, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        
                        fixed_count += 1
                        self.logger.info(f"🔧 Paths corrigidos em: {config_file.name}")
                
                except Exception as e:
                    self.errors.append(f"Erro ao corrigir {config_file}: {e}")
            
            self.logger.info(f"✅ Correção de paths concluída: {fixed_count} arquivos")
            return await True
            
        except Exception as e:
            self.logger.error(f"Erro na correção de paths: {e}")
            return await False
    
    async def _fix_paths_recursive(self, data: Dict) -> bool:
        """Corrige paths recursivamente em estrutura de dados."""
        changes_made = False
        
        for key, value in data.items():
            if isinstance(value, str):
                # Corrige paths conhecidos
                old_value = value
                
                # Substitui paths antigos por novos
                if "/root/penin_omega" in value:
                    value = value.replace("/root/penin_omega", str(PENIN_OMEGA_ROOT))
                    changes_made = True
                
                if "~/.penin_omega" in value:
                    value = value.replace("~/.penin_omega", str(PENIN_OMEGA_ROOT))
                    changes_made = True
                
                # Corrige paths relativos
                if value.startswith("./") and "penin_omega" in value:
                    value = str(PENIN_OMEGA_ROOT / value[2:])
                    changes_made = True
                
                if old_value != value:
                    data[key] = value
                    
            elif isinstance(value, dict):
                if self._fix_paths_recursive(value):
                    changes_made = True
                    
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        if self._fix_paths_recursive(item):
                            changes_made = True
        
        return await changes_made
    
    async def validate_file_structure(self) -> Dict[str, Any]:
        """Valida estrutura de arquivos do sistema."""
        validation_result = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            # Verifica diretórios essenciais
            essential_dirs = [
                "logs", "config", "cache", "artifacts", 
                "knowledge", "worm", "modules"
            ]
            
            missing_dirs = []
            for dir_name in essential_dirs:
                dir_path = PENIN_OMEGA_ROOT / dir_name
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                validation_result["issues"].append(f"Diretórios ausentes: {missing_dirs}")
                validation_result["valid"] = False
            
            # Estatísticas
            validation_result["statistics"] = {
                "total_files": len(list(PENIN_OMEGA_ROOT.rglob("*"))),
                "python_modules": len(list(PENIN_OMEGA_ROOT.rglob("*.py"))),
                "log_files": len(list(PENIN_OMEGA_ROOT.rglob("*.log"))),
                "config_files": len(list(PENIN_OMEGA_ROOT.rglob("*.json"))),
                "database_files": len(list(PENIN_OMEGA_ROOT.rglob("*.db"))),
                "duplicates_groups": len(self.duplicates_found)
            }
            
            self.logger.info(f"📊 Validação: {validation_result['statistics']}")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Erro na validação: {e}")
        
        return await validation_result
    
    async def generate_report(self) -> Dict[str, Any]:
        """Gera relatório completo da organização."""
        return await {
            "timestamp": datetime.now().isoformat(),
            "duplicates_found": len(self.duplicates_found),
            "files_deleted": len(self.files_deleted),
            "files_moved": len(self.files_moved),
            "errors": len(self.errors),
            "validation": self.validate_file_structure(),
            "details": {
                "deleted_files": self.files_deleted,
                "moved_files": self.files_moved,
                "errors": self.errors
            }
        }

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

async def organize_penin_omega_files():
    """Executa organização completa do sistema de arquivos."""
    organizer = FileOrganizer()
    
    print("🧹 Iniciando organização do sistema de arquivos...")
    
    # 1. Escaneia duplicatas
    duplicates = organizer.scan_duplicates()
    print(f"🔍 Duplicatas encontradas: {len(duplicates)} grupos")
    
    # 2. Remove duplicatas
    removed = organizer.clean_duplicates()
    print(f"🗑️  Arquivos removidos: {removed}")
    
    # 3. Organiza arquivos
    organized = organizer.organize_penin_files()
    if organized:
        print("📁 Arquivos organizados com sucesso")
    
    # 4. Corrige paths
    paths_fixed = organizer.fix_paths_in_configs()
    if paths_fixed:
        print("🔧 Paths corrigidos em configurações")
    
    # 5. Valida estrutura
    validation = organizer.validate_file_structure()
    if validation["valid"]:
        print("✅ Estrutura de arquivos válida")
    else:
        print(f"⚠️  Problemas encontrados: {validation['issues']}")
    
    # 6. Gera relatório
    report = organizer.generate_report()
    
    # Salva relatório
    report_file = PENIN_OMEGA_ROOT / "logs" / "file_organization_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Relatório salvo: {report_file}")
    print("🎉 Organização do sistema de arquivos concluída!")
    
    return await report

# =============================================================================
# TESTE
# =============================================================================

if __name__ == "__main__":
    organize_penin_omega_files()
