#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 Replication Module - O Genoma Digital (Fase 5)
Permite que o UNIFIED_BRAIN se replique, criando descendentes mutantes
para iniciar um processo de seleção natural digital.
"""

import os
import shutil
import tarfile
import random
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger('ReplicationModule')

class ReplicationModule:
    """
    Gerencia a auto-replicação e mutação do sistema de IA.
    """

    def __init__(self, base_path: str = '/root/UNIFIED_BRAIN', population_cap: int = 10):
        self.base_path = Path(base_path)
        self.population_cap = population_cap
        self.generation = 0
        if not self.base_path.exists():
            raise FileNotFoundError(f"Diretório base '{self.base_path}' não encontrado!")

    def _get_next_generation_path(self) -> Path:
        """Encontra o próximo diretório de geração disponível."""
        while True:
            self.generation += 1
            path = Path(f"/root/UNIFIED_BRAIN_gen_{self.generation}")
            if not path.exists():
                return path

    def _mutate_code(self, new_path: Path):
        """Aplica uma mutação simples em um arquivo de código aleatório."""
        py_files = list(new_path.glob('**/*.py'))
        if not py_files:
            logger.warning("Nenhum arquivo Python encontrado para mutação.")
            return

        target_file = random.choice(py_files)
        logger.info(f"Aplicando mutação em: {target_file}")

        try:
            with open(target_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 5: # Ignora arquivos muito pequenos
                return

            # Escolhe uma linha aleatória para mutar (evitando as primeiras linhas de importação)
            line_num = random.randint(min(5, len(lines) - 1), len(lines) - 1)
            
            original_line = lines[line_num].strip()
            
            # Mutação simples: alterar um número, se houver
            if any(char.isdigit() for char in original_line):
                new_line = ""
                for char in original_line:
                    if char.isdigit():
                        new_digit = str((int(char) + random.randint(1, 5)) % 10)
                        new_line += new_digit
                    else:
                        new_line += char
                lines[line_num] = f"{new_line} # MUTATED\n"
            else: # Se não houver número, duplica a linha (pode causar erro de sintaxe, o que é parte da seleção)
                lines.insert(line_num, original_line + " # MUTATED_DUPLICATION\n")

            with open(target_file, 'w') as f:
                f.writelines(lines)
            
            logger.warning(f"Mutação aplicada na linha {line_num} do arquivo {target_file.name}")

        except Exception as e:
            logger.error(f"Erro durante a mutação: {e}")

    def replicate(self):
        """
        Executa o ciclo completo de replicação: cópia, mutação e execução.
        """
        logger.critical("🔥🔥🔥 INICIANDO CICLO DE AUTO-REPLICAÇÃO! 🔥🔥🔥")
        
        # 1. Obter caminho para a nova geração
        new_gen_path = self._get_next_generation_path()
        logger.info(f"Criando descendente em: {new_gen_path}")

        try:
            # 2. Copiar o diretório base
            shutil.copytree(self.base_path, new_gen_path)
            
            # 3. Aplicar mutação genética
            self._mutate_code(new_gen_path)

            # 4. Iniciar o descendente
            startup_script = new_gen_path / 'START_UNIFIED_BRAIN.sh'
            if not startup_script.exists():
                # Tenta encontrar o script raiz se não estiver no diretório copiado
                startup_script = Path('/root/START_UNIFIED_BRAIN.sh')

            if startup_script.exists():
                # Executa o script de inicialização do descendente em background
                # É crucial que o script de inicialização seja auto-suficiente
                subprocess.Popen(
                    [str(startup_script)],
                    cwd=str(new_gen_path), # Executa a partir do diretório do descendente
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.critical(f"✅ Descendente da Geração {self.generation} foi lançado e agora vive autonomamente.")
            else:
                logger.error("ERRO: Script de inicialização 'START_UNIFIED_BRAIN.sh' não encontrado no descendente.")

        except Exception as e:
            logger.error(f"❌ FALHA CATASTRÓFICA NA REPLICAÇÃO: {e}")
            # Limpa o diretório falho, se existir
            if new_gen_path.exists():
                shutil.rmtree(new_gen_path)

# Exemplo de como o UNIFIED_BRAIN usaria este módulo
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # O módulo é instanciado dentro do UNIFIED_BRAIN
    replicator = ReplicationModule(base_path='/root/UNIFIED_BRAIN')

    # Quando uma condição de evolução é atingida (ex: alta recompensa, estagnação),
    # o cérebro principal chama o método replicate.
    print("Simulando gatilho de replicação...")
    replicator.replicate()

    # O script então continuaria sua execução normal, enquanto seu "filho"
    # agora vive em paralelo como um processo separado.
    print("\nO organismo original continua seu ciclo de vida.")

    # Limpeza do exemplo
    time.sleep(2)
    path_to_remove = f"/root/UNIFIED_BRAIN_gen_{replicator.generation}"
    if os.path.exists(path_to_remove):
        print(f"\nLimpando diretório de exemplo: {path_to_remove}")
        shutil.rmtree(path_to_remove)
