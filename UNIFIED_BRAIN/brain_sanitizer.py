#!/usr/bin/env python3
"""
🛡️ BRAIN SANITIZER
Detecta side-effects, network calls, execuções perigosas
"""

import ast
import os
import hashlib
import json
from typing import Dict, List, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class ScanResult:
    """Resultado do scan de um arquivo"""
    path: str
    checksum: str
    status: str  # 'safe', 'suspect', 'dangerous'
    findings: List[Dict[str, Any]]
    neuron_classes: List[str]
    imports: Set[str]
    has_forward: bool
    has_side_effects: bool
    
    def to_dict(self):
        d = asdict(self)
        d['imports'] = list(d['imports'])
        return d


class ASTSanitizer:
    """
    Scanner AST para detectar código perigoso
    """
    
    # Chamadas perigosas
    DANGEROUS_CALLS = {
        'eval', 'exec', 'compile',
        'os.system', 'os.popen', 'os.execv',
        'subprocess.Popen', 'subprocess.call', 'subprocess.run',
        'socket.socket', 'socket.create_connection',
        'requests.get', 'requests.post',
        'urllib.request', 'urllib.urlopen',
        'ftplib', 'paramiko',
        'shutil.rmtree',
        '__import__',
        'importlib.import_module',
    }
    
    # Imports suspeitos
    SUSPECT_IMPORTS = {
        'subprocess', 'socket', 'requests', 'urllib',
        'ftplib', 'paramiko', 'telnetlib', 'smtplib',
        'pickle',  # desserialização arbitrária
    }
    
    # Operações de escrita
    WRITE_OPERATIONS = {
        'open',  # se mode='w'
        'file.write',
        'os.remove', 'os.unlink',
        'shutil.rmtree', 'shutil.move',
    }
    
    def __init__(self):
        self.findings = []
        self.imports = set()
        self.neuron_classes = []
        self.has_forward = False
        self.has_side_effects = False
    
    def scan_file(self, filepath: str) -> ScanResult:
        """
        Scannea arquivo Python completo
        """
        self.findings = []
        self.imports = set()
        self.neuron_classes = []
        self.has_forward = False
        self.has_side_effects = False
        
        # Checksum
        with open(filepath, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        # Parse AST
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception as e:
            return ScanResult(
                path=filepath,
                checksum=checksum,
                status='error',
                findings=[{'error': str(e)}],
                neuron_classes=[],
                imports=set(),
                has_forward=False,
                has_side_effects=False
            )
        
        # Visitor
        visitor = self._ScanVisitor(self)
        visitor.visit(tree)
        
        # Determina status
        status = self._determine_status()
        
        return ScanResult(
            path=filepath,
            checksum=checksum,
            status=status,
            findings=self.findings,
            neuron_classes=self.neuron_classes,
            imports=self.imports,
            has_forward=self.has_forward,
            has_side_effects=self.has_side_effects
        )
    
    def _determine_status(self) -> str:
        """Classifica arquivo como safe/suspect/dangerous"""
        if any(f.get('severity') == 'critical' for f in self.findings):
            return 'dangerous'
        elif self.has_side_effects or len(self.findings) > 0:
            return 'suspect'
        else:
            return 'safe'
    
    class _ScanVisitor(ast.NodeVisitor):
        """Visitor interno para AST"""
        def __init__(self, scanner):
            self.scanner = scanner
        
        def visit_Import(self, node):
            for alias in node.names:
                self.scanner.imports.add(alias.name)
                if alias.name in ASTSanitizer.SUSPECT_IMPORTS:
                    self.scanner.findings.append({
                        'type': 'suspect_import',
                        'lineno': node.lineno,
                        'module': alias.name,
                        'severity': 'medium'
                    })
            self.generic_visit(node)
        
        def visit_ImportFrom(self, node):
            if node.module:
                self.scanner.imports.add(node.module)
                if node.module in ASTSanitizer.SUSPECT_IMPORTS:
                    self.scanner.findings.append({
                        'type': 'suspect_import',
                        'lineno': node.lineno,
                        'module': node.module,
                        'severity': 'medium'
                    })
            self.generic_visit(node)
        
        def visit_ClassDef(self, node):
            # Detecta classes de neurônios
            if 'neuron' in node.name.lower() or any(
                hasattr(b, 'id') and 'neuron' in getattr(b, 'id', '').lower()
                for b in node.bases
            ):
                self.scanner.neuron_classes.append(node.name)
            
            # Verifica se tem método forward
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                    self.scanner.has_forward = True
            
            self.generic_visit(node)
        
        def visit_Call(self, node):
            # Detecta chamadas perigosas
            try:
                func_name = ast.unparse(node.func)
            except:
                func_name = getattr(node.func, 'id', str(type(node.func)))
            
            # Verifica contra lista de perigosos
            for dangerous in ASTSanitizer.DANGEROUS_CALLS:
                if dangerous in func_name:
                    self.scanner.has_side_effects = True
                    self.scanner.findings.append({
                        'type': 'dangerous_call',
                        'lineno': node.lineno,
                        'call': func_name,
                        'severity': 'critical'
                    })
                    break
            
            # Detecta open() com modo write
            if 'open' in func_name:
                for kw in node.keywords:
                    if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                        if 'w' in str(kw.value.value) or 'a' in str(kw.value.value):
                            self.scanner.has_side_effects = True
                            self.scanner.findings.append({
                                'type': 'file_write',
                                'lineno': node.lineno,
                                'call': func_name,
                                'severity': 'high'
                            })
            
            self.generic_visit(node)


class QuarantineManager:
    """
    Gerencia neurônios em quarentena
    """
    def __init__(self, quarantine_dir: str = "/root/UNIFIED_BRAIN/quarantine"):
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.quarantine_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def quarantine_file(self, scan_result: ScanResult):
        """Move arquivo suspeito para quarentena"""
        # Cria subdir por severidade
        reason = scan_result.status
        reason_dir = self.quarantine_dir / reason
        reason_dir.mkdir(exist_ok=True)
        
        # Copia arquivo
        src = Path(scan_result.path)
        dst = reason_dir / f"{scan_result.checksum[:16]}_{src.name}"
        
        import shutil
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        
        # Atualiza manifest
        self.manifest[scan_result.checksum] = {
            'original_path': scan_result.path,
            'quarantine_path': str(dst),
            'status': scan_result.status,
            'findings': scan_result.findings,
            'timestamp': str(Path(scan_result.path).stat().st_mtime),
            'reviewed': False,
            'reviewer': None
        }
        self._save_manifest()
    
    def approve(self, checksum: str, reviewer: str):
        """Aprova neurônio para sair da quarentena"""
        if checksum in self.manifest:
            self.manifest[checksum]['reviewed'] = True
            self.manifest[checksum]['reviewer'] = reviewer
            self.manifest[checksum]['status'] = 'approved'
            self._save_manifest()
    
    def get_pending_review(self) -> List[Dict]:
        """Retorna lista de arquivos aguardando revisão"""
        return [
            {**item, 'checksum': csum}
            for csum, item in self.manifest.items()
            if not item.get('reviewed', False)
        ]


def scan_directory(directory: str, output_file: str = None) -> List[ScanResult]:
    """
    Scannea diretório completo
    """
    sanitizer = ASTSanitizer()
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certos dirs
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.cache']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    result = sanitizer.scan_file(filepath)
                    results.append(result)
                except Exception as e:
                    print(f"ERROR scanning {filepath}: {e}")
    
    # Salva resultados
    if output_file:
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    
    return results


if __name__ == "__main__":
    print("🛡️ Brain Sanitizer Module")
    print("Run: python brain_sanitizer.py /path/to/scan")
    
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        print(f"\nScanning: {directory}")
        results = scan_directory(directory, "scan_results.json")
        
        safe = sum(1 for r in results if r.status == 'safe')
        suspect = sum(1 for r in results if r.status == 'suspect')
        dangerous = sum(1 for r in results if r.status == 'dangerous')
        
        print(f"\n✅ Safe: {safe}")
        print(f"⚠️  Suspect: {suspect}")
        print(f"🚨 Dangerous: {dangerous}")
        print(f"\nResults saved to: scan_results.json")
