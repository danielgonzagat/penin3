#!/usr/bin/env python3
# PENIN-Ω Auto-Inspection: Análise recursiva do código para auto-melhoria
# Usa AST para identificar problemas e propor evoluções estruturais.

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

ROOT = Path('/root/.penin_omega')
MODULES_DIR = ROOT / 'modules'
LOG = ROOT / 'logs' / 'auto_inspection.log'


async def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{time.time():.0f}] {msg}\n")


class CodeInspector(ast.NodeVisitor):
    """Inspeciona código Python com AST para identificar problemas e oportunidades de melhoria."""

    async def __init__(self, source_file: str):
        self.source_file = source_file
        self.issues: List[Dict[str, Any]] = []
        self.complexity_score = 0
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        self.loop_depth = 0
        self.max_loop_depth = 0

    async def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        # Analisa complexidade da função
        complexity = self._calculate_complexity(node)
        if complexity > 10:
            self.issues.append({
                "type": "high_complexity",
                "location": f"{self.source_file}:{node.lineno}",
                "function": node.name,
                "complexity": complexity,
                "suggestion": "Refatorar função em subfunções menores"
            })
        self.generic_visit(node)

    async def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_count += 1
        self.generic_visit(node)

    async def visit_Import(self, node: ast.Import) -> None:
        self.import_count += 1
        self.generic_visit(node)

    async def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_count += 1
        self.generic_visit(node)

    async def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        if self.loop_depth > 3:
            self.issues.append({
                "type": "deep_nesting",
                "location": f"{self.source_file}:{node.lineno}",
                "suggestion": "Reduzir profundidade de loops aninhados"
            })
        self.generic_visit(node)
        self.loop_depth -= 1

    async def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        # Verifica loops infinitos potenciais
        if not node.test or isinstance(node.test, ast.NameConstant) and node.test.value:
            self.issues.append({
                "type": "potential_infinite_loop",
                "location": f"{self.source_file}:{node.lineno}",
                "suggestion": "Adicionar condição de parada clara"
            })
        self.generic_visit(node)
        self.loop_depth -= 1

    async def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcula complexidade ciclomática aproximada."""
        complexity = 1  # Base
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
        return await complexity

    async def get_inspection_report(self) -> Dict[str, Any]:
        return await {
            "file": self.source_file,
            "stats": {
                "functions": self.function_count,
                "classes": self.class_count,
                "imports": self.import_count,
                "max_loop_depth": self.max_loop_depth,
                "total_issues": len(self.issues)
            },
            "issues": self.issues
        }


async def inspect_file(file_path: Path) -> Dict[str, Any]:
    """Inspeciona um arquivo Python."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        inspector = CodeInspector(str(file_path))
        inspector.visit(tree)
        return await inspector.get_inspection_report()

    except Exception as e:
        return await {
            "file": str(file_path),
            "error": str(e),
            "stats": {},
            "issues": []
        }


async def inspect_system() -> Dict[str, Any]:
    """Inspeciona todo o sistema PENIN-Ω."""
    reports = []
    total_issues = 0

    # Inspeciona módulos principais
    for py_file in MODULES_DIR.glob("*.py"):
        report = inspect_file(py_file)
        reports.append(report)
        total_issues += len(report.get("issues", []))

    # Identifica arquivos com mais problemas
    problematic_files = sorted(
        reports,
        key=lambda r: len(r.get("issues", [])),
        reverse=True
    )[:3]

    # Gera sugestões de evolução estrutural
    suggestions = []
    if total_issues > 10:
        suggestions.append("Refatorar módulos com alta complexidade")
    if any(r.get("stats", {}).get("max_loop_depth", 0) > 3 for r in reports):
        suggestions.append("Reduzir profundidade de aninhamento em loops")
    if total_issues > 20:
        suggestions.append("Implementar arquitetura modular mais granular")

    return await {
        "timestamp": time.time(),
        "total_files": len(reports),
        "total_issues": total_issues,
        "problematic_files": problematic_files,
        "structural_suggestions": suggestions,
        "reports": reports
    }


async def propose_structural_improvements(inspection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Propõe melhorias estruturais baseadas na inspeção."""
    improvements = []

    # Sugestão baseada em issues críticas
    critical_issues = sum(
        len(r.get("issues", [])) for r in inspection.get("reports", [])
        if any(iss.get("type") in ["high_complexity", "deep_nesting"] for iss in r.get("issues", []))
    )

    if critical_issues > 5:
        improvements.append({
            "type": "refactoring",
            "description": "Refatorar funções complexas em módulos menores",
            "impact": "high",
            "files_affected": [r["file"] for r in inspection.get("reports", []) if r.get("issues")]
        })

    # Sugestão de otimização de performance
    deep_loops = any(
        r.get("stats", {}).get("max_loop_depth", 0) > 3
        for r in inspection.get("reports", [])
    )
    if deep_loops:
        improvements.append({
            "type": "optimization",
            "description": "Otimizar estruturas de loop profundas",
            "impact": "medium",
            "files_affected": ["penin_behavior_harness.py", "penin_unified_bridge.py"]
        })

    # Sugestão de arquitetura
    if inspection.get("total_issues", 0) > 30:
        improvements.append({
            "type": "architecture",
            "description": "Implementar arquitetura baseada em agentes especializados",
            "impact": "high",
            "files_affected": ["all_modules"]
        })

    return await improvements


async def auto_inspect_and_propose() -> Dict[str, Any]:
    """Executa inspeção completa e propõe melhorias."""
    inspection = inspect_system()
    improvements = propose_structural_improvements(inspection)

    result = {
        "inspection": inspection,
        "proposed_improvements": improvements,
        "actionable": len(improvements) > 0
    }

    log(f"Auto-inspection completed: {inspection['total_issues']} issues found, {len(improvements)} improvements proposed")
    return await result


if __name__ == "__main__":
    result = auto_inspect_and_propose()
    logger.info(json.dumps(result, indent=2, ensure_ascii=False))
