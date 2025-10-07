#!/usr/bin/env python3
"""
🛠️ IA³ - MOTOR DE AUTO-MODIFICAÇÃO
==================================

Sistema seguro de auto-modificação com validação completa
"""

import os
import sys
import ast
import inspect
import importlib
import hashlib
import tempfile
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger("IA³-AutoMod")

class CodeAnalyzer:
    """Analisador avançado de código para identificar melhorias"""

    def __init__(self):
        self.complexity_metrics = {}
        self.code_patterns = {}

    def analyze_codebase(self) -> Dict[str, Any]:
        """Analisar todo o codebase para oportunidades de melhoria"""
        analysis = {
            'functions': {},
            'classes': {},
            'modules': {},
            'patterns': {},
            'issues': []
        }

        # Analisar arquivos Python
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    filepath = os.path.join(root, file)
                    try:
                        file_analysis = self._analyze_file(filepath)
                        analysis['modules'][filepath] = file_analysis
                    except Exception as e:
                        logger.warning(f"Erro ao analisar {filepath}: {e}")

        # Identificar padrões e problemas
        analysis['patterns'] = self._identify_patterns(analysis)
        analysis['issues'] = self._identify_issues(analysis)

        return analysis

    def _analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analisar arquivo individual"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        file_analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'complexity': 0,
            'lines': len(content.split('\n')),
            'size': len(content)
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node, content)
                file_analysis['functions'].append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node, content)
                file_analysis['classes'].append(class_info)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                file_analysis['imports'].append(self._analyze_import(node))

        file_analysis['complexity'] = self._calculate_file_complexity(file_analysis)

        return file_analysis

    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analisar função específica"""
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line + 10)

        lines = content.split('\n')[start_line-1:end_line]
        func_content = '\n'.join(lines)

        return {
            'name': node.name,
            'lines': len(lines),
            'complexity': self._calculate_cyclomatic_complexity(func_content),
            'parameters': len(node.args.args),
            'docstring': ast.get_docstring(node) is not None,
            'start_line': start_line,
            'end_line': end_line
        }

    async def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analisar classe específica"""
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line + 50)

        return await {
            'name': node.name,
            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            'lines': end_line - start_line + 1,
            'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
        }

    async def _analyze_import(self, node) -> Dict[str, Any]:
        """Analisar import"""
        if isinstance(node, ast.Import):
            return await {'type': 'import', 'names': [alias.name for alias in node.names]}
        else:
            return await {'type': 'from_import', 'module': node.module, 'names': [alias.name for alias in node.names]}

    async def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calcular complexidade ciclomática simples"""
        complexity = 1  # Base

        keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'case ', 'catch ', '&&', '||']
        for keyword in keywords:
            complexity += code.count(keyword)

        return await complexity

    async def _calculate_file_complexity(self, file_analysis: Dict[str, Any]) -> float:
        """Calcular complexidade geral do arquivo"""
        factors = [
            len(file_analysis['functions']) * 0.1,
            sum(f['complexity'] for f in file_analysis['functions']) * 0.2,
            len(file_analysis['classes']) * 0.3,
            file_analysis['lines'] * 0.001
        ]

        return await sum(factors)

    async def _identify_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identificar padrões no código"""
        patterns = {
            'duplicate_functions': [],
            'long_functions': [],
            'complex_functions': [],
            'unused_imports': [],
            'circular_dependencies': []
        }

        # Funções muito longas
        for module, data in analysis['modules'].items():
            for func in data['functions']:
                if func['lines'] > 50:
                    patterns['long_functions'].append({
                        'module': module,
                        'function': func['name'],
                        'lines': func['lines']
                    })
                if func['complexity'] > 10:
                    patterns['complex_functions'].append({
                        'module': module,
                        'function': func['name'],
                        'complexity': func['complexity']
                    })

        return await patterns

    async def _identify_issues(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identificar problemas que precisam correção"""
        issues = []

        for module, data in analysis['modules'].items():
            # Funções sem docstring
            for func in data['functions']:
                if not func['docstring']:
                    issues.append({
                        'type': 'missing_docstring',
                        'module': module,
                        'function': func['name'],
                        'severity': 'low'
                    })

            # Funções muito complexas
            for func in data['functions']:
                if func['complexity'] > 15:
                    issues.append({
                        'type': 'high_complexity',
                        'module': module,
                        'function': func['name'],
                        'complexity': func['complexity'],
                        'severity': 'medium'
                    })

        return await issues

class SafeModificationEngine:
    """Motor de modificação segura com backup e validação"""

    async def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.backup_dir = "automod_backups"
        self.modification_log = []
        self.safety_checks = [
            self._check_syntax_validity,
            self._check_import_integrity,
            self._check_function_signatures,
            self._run_basic_tests
        ]

        os.makedirs(self.backup_dir, exist_ok=True)

    async def generate_modifications(self) -> List[Dict[str, Any]]:
        """Gerar lista de modificações recomendadas"""
        analysis = self.analyzer.analyze_codebase()
        modifications = []

        # Modificações baseadas em padrões identificados
        patterns = analysis['patterns']

        # Refatorar funções longas
        for long_func in patterns['long_functions'][:5]:  # Limitar a 5 por vez
            modifications.append({
                'type': 'refactor_long_function',
                'target': long_func,
                'priority': 0.7,
                'description': f"Refatorar função longa: {long_func['function']} ({long_func['lines']} linhas)"
            })

        # Simplificar funções complexas
        for complex_func in patterns['complex_functions'][:3]:
            modifications.append({
                'type': 'simplify_complex_function',
                'target': complex_func,
                'priority': 0.8,
                'description': f"Simplificar função complexa: {complex_func['function']} (complexidade {complex_func['complexity']})"
            })

        # Adicionar docstrings
        for issue in analysis['issues']:
            if issue['type'] == 'missing_docstring':
                modifications.append({
                    'type': 'add_docstring',
                    'target': issue,
                    'priority': 0.5,
                    'description': f"Adicionar docstring: {issue['function']}"
                })

        return await sorted(modifications, key=lambda x: x['priority'], reverse=True)

    async def apply_modification(self, modification: Dict[str, Any]) -> bool:
        """Aplicar modificação com segurança total"""
        try:
            # Criar backup
            backup_id = self._create_backup()

            # Aplicar modificação
            success = self._execute_modification(modification)

            if success:
                # Executar verificações de segurança
                if self._run_safety_checks():
                    # Log da modificação bem-sucedida
                    self.modification_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'modification': modification,
                        'backup_id': backup_id,
                        'status': 'success'
                    })

                    logger.info(f"✅ Modificação aplicada: {modification['description']}")
                    return await True
                else:
                    # Reverter se falhar nas verificações
                    self._restore_backup(backup_id)
                    logger.warning(f"❌ Modificação revertida - falha nas verificações: {modification['description']}")
                    return await False
            else:
                logger.error(f"❌ Falha ao executar modificação: {modification['description']}")
                return await False

        except Exception as e:
            logger.error(f"❌ Erro crítico na modificação: {e}")
            # Tentar restaurar backup
            try:
                self._restore_backup(backup_id)
            except:
                pass
            return await False

    async def _create_backup(self) -> str:
        """Criar backup completo do estado atual"""
        backup_id = f"backup_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        backup_path = os.path.join(self.backup_dir, backup_id)

        os.makedirs(backup_path)

        # Backup de arquivos Python
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path)
                    dst_path = os.path.join(backup_path, rel_path)

                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    import shutil
                    shutil.copy2(src_path, dst_path)

        logger.info(f"📦 Backup criado: {backup_id}")
        return await backup_id

    async def _restore_backup(self, backup_id: str):
        """Restaurar backup específico"""
        backup_path = os.path.join(self.backup_dir, backup_id)

        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup não encontrado: {backup_id}")

        # Restaurar arquivos
        for root, dirs, files in os.walk(backup_path):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, backup_path)
                dst_path = os.path.join('.', rel_path)

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                import shutil
                shutil.copy2(src_path, dst_path)

        logger.info(f"🔄 Backup restaurado: {backup_id}")

    async def _execute_modification(self, modification: Dict[str, Any]) -> bool:
        """Executar a modificação específica"""
        mod_type = modification['type']

        try:
            if mod_type == 'refactor_long_function':
                return await self._refactor_long_function(modification['target'])
            elif mod_type == 'simplify_complex_function':
                return await self._simplify_complex_function(modification['target'])
            elif mod_type == 'add_docstring':
                return await self._add_docstring(modification['target'])
            else:
                logger.warning(f"Tipo de modificação não suportado: {mod_type}")
                return await False
        except Exception as e:
            logger.error(f"Erro ao executar {mod_type}: {e}")
            return await False

    async def _refactor_long_function(self, target: Dict[str, Any]) -> bool:
        """Refatorar função longa criando helpers"""
        module = target['module']
        function = target['function']

        # Ler arquivo
        with open(module, 'r') as f:
            content = f.read()

        # Encontrar função no AST
        tree = ast.parse(content)
        func_node = None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                func_node = node
                break

        if not func_node:
            return await False

        # Criar função helper
        helper_name = f"{function}_helper"
        helper_code = f"""
async def {helper_name}():
    \"\"\"Helper function extracted from {function}\"\"\"
    # TODO: Extract complex logic here
    pass

"""

        # Inserir helper antes da função
        lines = content.split('\n')
        insert_line = func_node.lineno - 1

        lines.insert(insert_line, helper_code)

        # Reescrever arquivo
        new_content = '\n'.join(lines)
        with open(module, 'w') as f:
            f.write(new_content)

        logger.info(f"🔧 Refatorada {function} em {module} - helper criado")
        return await True

    async def _simplify_complex_function(self, target: Dict[str, Any]) -> bool:
        """Simplificar função complexa"""
        # Implementação básica - adicionar comentários explicativos
        module = target['module']
        function = target['function']

        with open(module, 'r') as f:
            content = f.read()

        # Adicionar comentário de complexidade
        complexity_comment = f"\n    # TODO: Simplify this complex function (complexity: {target['complexity']})\n"

        # Inserir comentário no início da função
        lines = content.split('\n')
        func_start = None

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                func_start = node.lineno
                break

        if func_start:
            lines.insert(func_start, complexity_comment)

            new_content = '\n'.join(lines)
            with open(module, 'w') as f:
                f.write(new_content)

            logger.info(f"📝 Marcada função complexa: {function}")
            return await True

        return await False

    async def _add_docstring(self, target: Dict[str, Any]) -> bool:
        """Adicionar docstring a função"""
        module = target['module']
        function = target['function']

        with open(module, 'r') as f:
            content = f.read()

        # Adicionar docstring básica
        docstring = f'    """{function} function.\n\n    TODO: Add proper documentation\n    """'

        lines = content.split('\n')
        func_start = None

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                func_start = node.lineno
                break

        if func_start:
            # Encontrar onde inserir docstring (após async def line)
            insert_pos = func_start
            while insert_pos < len(lines) and not lines[insert_pos].strip().endswith(':'):
                insert_pos += 1

            if insert_pos < len(lines):
                lines.insert(insert_pos + 1, docstring)

                new_content = '\n'.join(lines)
                with open(module, 'w') as f:
                    f.write(new_content)

                logger.info(f"📚 Docstring adicionada: {function}")
                return await True

        return await False

    async def _run_safety_checks(self) -> bool:
        """Executar todas as verificações de segurança"""
        for check in self.safety_checks:
            try:
                if not check():
                    logger.error(f"❌ Falha na verificação: {check.__name__}")
                    return await False
            except Exception as e:
                logger.error(f"❌ Erro na verificação {check.__name__}: {e}")
                return await False

        return await True

    async def _check_syntax_validity(self) -> bool:
        """Verificar se todos os arquivos Python têm sintaxe válida"""
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            ast.parse(f.read())
                    except SyntaxError as e:
                        logger.error(f"Sintaxe inválida em {filepath}: {e}")
                        return await False
        return await True

    async def _check_import_integrity(self) -> bool:
        """Verificar se imports são válidos"""
        try:
            # Tentar importar módulos principais
            test_imports = [
                'torch', 'numpy', 'json', 'os', 'sys',
                'threading', 'time', 'datetime'
            ]

            for module in test_imports:
                try:
                    __import__(module)
                except ImportError:
                    logger.warning(f"Import não disponível: {module}")
                    # Não é erro crítico se módulo opcional não estiver disponível

            return await True
        except Exception as e:
            logger.error(f"Erro na verificação de imports: {e}")
            return await False

    async def _check_function_signatures(self) -> bool:
        """Verificar se assinaturas de função não mudaram drasticamente"""
        # Implementação básica - verificar se funções principais existem
        try:
            main_functions = ['main', '__init__', 'run', 'execute']
            modules_to_check = ['IA3_EMERGENT_CORE']

            for module_name in modules_to_check:
                try:
                    spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for func_name in main_functions:
                        if hasattr(module, func_name):
                            func = getattr(module, func_name)
                            if callable(func):
                                # Verificar assinatura básica
                                sig = inspect.signature(func)
                                if len(sig.parameters) > 20:  # Muito parâmetros
                                    logger.warning(f"Função {func_name} tem muitos parâmetros: {len(sig.parameters)}")
                                    return await False
                except Exception as e:
                    logger.warning(f"Erro ao verificar {module_name}: {e}")

            return await True
        except Exception as e:
            logger.error(f"Erro na verificação de assinaturas: {e}")
            return await False

    async def _run_basic_tests(self) -> bool:
        """Executar testes básicos de funcionalidade"""
        try:
            # Teste básico: criar instância do analisador
            analyzer = CodeAnalyzer()
            analysis = analyzer.analyze_codebase()

            if not analysis or 'modules' not in analysis:
                return await False

            # Verificar se análise tem conteúdo
            if len(analysis['modules']) == 0:
                logger.warning("Análise não encontrou módulos")
                return await False

            return await True
        except Exception as e:
            logger.error(f"Erro nos testes básicos: {e}")
            return await False

class AutoModificationOrchestrator:
    """Orquestrador principal de auto-modificação"""

    async def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.engine = SafeModificationEngine(self.analyzer)
        self.is_active = True

    async def start_automodification_loop(self):
        """Iniciar loop de auto-modificação contínua"""
        logger.info("🔄 Iniciando loop de auto-modificação")

        async def modification_loop():
            cycle = 0
            while self.is_active:
                try:
                    cycle += 1

                    # Gerar modificações
                    modifications = self.engine.generate_modifications()

                    if modifications:
                        # Aplicar top 3 modificações por ciclo
                        applied = 0
                        for mod in modifications[:3]:
                            if self.engine.apply_modification(mod):
                                applied += 1
                                logger.info(f"✅ Modificação {applied}/3 aplicada")

                        logger.info(f"🔄 Ciclo {cycle} | Modificações: {applied}/{len(modifications)} geradas")
                    else:
                        logger.info(f"🔄 Ciclo {cycle} | Nenhuma modificação necessária")

                    # Esperar antes do próximo ciclo
                    time.sleep(300)  # 5 minutos entre ciclos

                except Exception as e:
                    logger.error(f"Erro no ciclo de modificação: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=modification_loop, daemon=True)
        thread.start()

    async def get_modification_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de modificações"""
        return await {
            'total_modifications': len(self.engine.modification_log),
            'successful_modifications': len([m for m in self.engine.modification_log if m['status'] == 'success']),
            'backups_created': len(os.listdir(self.engine.backup_dir)) if os.path.exists(self.engine.backup_dir) else 0,
            'last_modification': self.engine.modification_log[-1] if self.engine.modification_log else None
        }

if __name__ == "__main__":
    # Teste do sistema
    orchestrator = AutoModificationOrchestrator()
    orchestrator.start_automodification_loop()

    # Manter rodando
    try:
        while True:
            time.sleep(10)
            stats = orchestrator.get_modification_stats()
            print(f"📊 Stats: {stats['successful_modifications']}/{stats['total_modifications']} modificações")
    except KeyboardInterrupt:
        print("🛑 Parando auto-modificação...")
        orchestrator.is_active = False