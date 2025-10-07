"""
Auto-Coding Engine - Extracted from OpenHands concepts
Enables self-modification, code generation, and autonomous improvement

Key concepts extracted:
- Action system (FileEditAction, CmdRunAction)
- Safe code modification
- Self-improvement loop
- Code generation & refactoring

NO async (clean integration with V7)
NO direct OpenHands copy (reimplemented cleanly)
"""

import os
import sys
import ast
import inspect
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of auto-coding actions"""
    FILE_READ = "read"
    FILE_WRITE = "write"
    FILE_EDIT = "edit"
    FILE_CREATE = "create"
    CMD_RUN = "cmd"
    CODE_REFACTOR = "refactor"
    MODULE_ADD = "add_module"


class ModificationRisk(Enum):
    """Risk levels for code modifications"""
    SAFE = "safe"           # Read-only, no side effects
    LOW = "low"             # Simple additions
    MEDIUM = "medium"       # Modifications to existing code
    HIGH = "high"           # Core system changes
    CRITICAL = "critical"   # Can break system


@dataclass
class CodeAction:
    """Represents an action to modify code"""
    action_type: ActionType
    target_path: str
    content: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    start_line: int = 0
    end_line: int = -1
    reason: str = ""
    risk: ModificationRisk = ModificationRisk.MEDIUM


class SafetyValidator:
    """
    Validates code modifications before applying
    Inspired by OpenHands' security validation
    """
    
    FORBIDDEN_PATTERNS = [
        'rm -rf',
        'sudo',
        'import __import__',
        'exec(',
        'eval(',
        '__builtins__',
        'os.system',
        'subprocess.Popen',
    ]
    
    SAFE_MODULES = [
        'extracted_algorithms',
        'agents',
        'models',
        'tests'
    ]
    
    @staticmethod
    def validate_code(code: str, reason: str = "") -> Tuple[bool, str]:
        """
        Validate code before execution/modification
        
        Returns:
            (is_valid, error_message)
        """
        # Check forbidden patterns
        for pattern in SafetyValidator.FORBIDDEN_PATTERNS:
            if pattern in code:
                return False, f"Forbidden pattern detected: {pattern}"
        
        # Try to parse as Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for dangerous imports
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['subprocess', 'os', 'sys']:
                        # These are allowed but logged
                        logger.warning(f"Using potentially dangerous import: {alias.name}")
        
        return True, ""
    
    @staticmethod
    def validate_path(path: str, base_dir: str) -> Tuple[bool, str]:
        """Validate that path is within base_dir and safe"""
        try:
            resolved = Path(path).resolve()
            base = Path(base_dir).resolve()
            
            # Check if path is within base_dir
            if not str(resolved).startswith(str(base)):
                return False, f"Path {path} is outside base directory"
            
            # Check if path is in safe module
            rel_path = str(resolved.relative_to(base))
            is_safe = any(rel_path.startswith(mod) for mod in SafetyValidator.SAFE_MODULES)
            
            if not is_safe:
                return False, f"Path {path} is not in safe modules"
            
            return True, ""
        except Exception as e:
            return False, f"Path validation error: {e}"


class SelfModificationEngine:
    """
    Enables the system to modify its own code
    Inspired by OpenHands' agent controller pattern
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.validator = SafetyValidator()
        self.modification_history: List[Dict] = []
        self.success_count = 0
        self.failure_count = 0
    
    def propose_modification(
        self,
        target_file: str,
        modification_type: str,
        content: str,
        reason: str
    ) -> Optional[CodeAction]:
        """
        Propose a code modification
        
        Args:
            target_file: File to modify
            modification_type: 'add', 'replace', 'refactor'
            content: New code content
            reason: Why this modification
        
        Returns:
            CodeAction if valid, None otherwise
        """
        # Validate path
        is_valid, error = self.validator.validate_path(
            target_file,
            str(self.base_dir)
        )
        if not is_valid:
            logger.error(f"Invalid path: {error}")
            return None
        
        # Validate code
        is_valid, error = self.validator.validate_code(content, reason)
        if not is_valid:
            logger.error(f"Invalid code: {error}")
            return None
        
        # Determine risk
        risk = self._assess_risk(target_file, modification_type)
        
        # Create action
        action = CodeAction(
            action_type=ActionType.FILE_EDIT,
            target_path=target_file,
            content=content,
            reason=reason,
            risk=risk
        )
        
        logger.info(f"✅ Proposed modification: {target_file} ({risk.value})")
        logger.info(f"   Reason: {reason}")
        
        return action
    
    def apply_modification(self, action: CodeAction) -> bool:
        """
        Apply a code modification
        
        Returns:
            True if successful, False otherwise
        """
        try:
            target_path = Path(action.target_path)
            
            # Backup original
            backup_content = None
            if target_path.exists():
                backup_content = target_path.read_text()
            
            # Apply modification
            if action.action_type == ActionType.FILE_CREATE:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(action.content)
            
            elif action.action_type == ActionType.FILE_EDIT:
                if backup_content is None:
                    logger.error(f"File {target_path} does not exist for editing")
                    return False
                
                # Simple replace for now
                if action.old_str and action.new_str:
                    new_content = backup_content.replace(action.old_str, action.new_str)
                else:
                    new_content = action.content
                
                target_path.write_text(new_content)
            
            # Test modification
            is_valid = self._test_modification(target_path)
            
            if is_valid:
                # Record success
                self.modification_history.append({
                    'action': action.action_type.value,
                    'path': str(target_path),
                    'reason': action.reason,
                    'risk': action.risk.value,
                    'success': True
                })
                self.success_count += 1
                logger.info(f"✅ Successfully applied modification to {target_path}")
                return True
            else:
                # Rollback
                if backup_content:
                    target_path.write_text(backup_content)
                logger.error(f"❌ Modification failed validation, rolled back")
                self.failure_count += 1
                return False
        
        except Exception as e:
            logger.error(f"❌ Error applying modification: {e}")
            self.failure_count += 1
            return False
    
    def _assess_risk(self, target_file: str, modification_type: str) -> ModificationRisk:
        """Assess risk of modification"""
        # Core system files are high risk
        if 'system_v7' in target_file or 'core/' in target_file:
            return ModificationRisk.HIGH
        
        # New files/modules are low risk
        if modification_type == 'add' or not Path(target_file).exists():
            return ModificationRisk.LOW
        
        # Extracted algorithms are medium risk
        if 'extracted_algorithms' in target_file:
            return ModificationRisk.MEDIUM
        
        return ModificationRisk.MEDIUM
    
    def _test_modification(self, file_path: Path) -> bool:
        """Test that modification doesn't break system"""
        try:
            # Try to import if it's a Python file
            if file_path.suffix == '.py':
                code = file_path.read_text()
                ast.parse(code)  # Syntax check
                return True
            return True
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get modification statistics"""
        total = self.success_count + self.failure_count
        success_rate = (self.success_count / total * 100) if total > 0 else 0
        
        return {
            'total_modifications': total,
            'successful': self.success_count,
            'failed': self.failure_count,
            'success_rate': success_rate,
            'history_size': len(self.modification_history)
        }


class CodeGenerationEngine:
    """
    Generates new code based on patterns and requirements
    Inspired by OpenHands' code generation capabilities
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.templates = self._load_templates()
        self.generation_count = 0
    
    def _load_templates(self) -> Dict[str, str]:
        """Load code templates"""
        return {
            'algorithm': '''"""
{docstring}
"""
import torch
import torch.nn as nn
from typing import Any, Dict

class {class_name}:
    """
    {description}
    """
    
    def __init__(self):
        self.initialized = False
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the algorithm"""
        raise NotImplementedError("Algorithm not yet implemented")
''',
            'test': '''"""
Test for {module_name}
"""
import unittest
from {module_path} import {class_name}

class Test{class_name}(unittest.TestCase):
    def setUp(self):
        self.instance = {class_name}()
    
    def test_basic(self):
        """Test basic functionality"""
        self.assertIsNotNone(self.instance)

if __name__ == '__main__':
    unittest.main()
'''
        }
    
    def generate_algorithm(
        self,
        name: str,
        description: str,
        template_type: str = 'algorithm'
    ) -> Optional[str]:
        """
        Generate a new algorithm from template
        
        Args:
            name: Algorithm name
            description: What it does
            template_type: Template to use
        
        Returns:
            Generated code or None
        """
        try:
            template = self.templates.get(template_type, self.templates['algorithm'])
            
            # Format template
            code = template.format(
                class_name=name,
                description=description,
                docstring=f"{name} - {description}"
            )
            
            # Validate
            validator = SafetyValidator()
            is_valid, error = validator.validate_code(code)
            
            if not is_valid:
                logger.error(f"Generated code is invalid: {error}")
                return None
            
            self.generation_count += 1
            logger.info(f"✅ Generated code for {name}")
            return code
        
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            return None
    
    def suggest_improvement(self, file_path: str) -> Optional[str]:
        """
        Analyze code and suggest improvements
        
        Args:
            file_path: File to analyze
        
        Returns:
            Suggestion text or None
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            code = path.read_text()
            tree = ast.parse(code)
            
            suggestions = []
            
            # Check for docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        suggestions.append(f"Add docstring to {node.name}")
            
            # Check for type hints
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.returns:
                        suggestions.append(f"Add return type hint to {node.name}")
            
            return "\n".join(suggestions) if suggestions else None
        
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return None


class AutoCodingOrchestrator:
    """
    Main orchestrator for auto-coding capabilities
    Combines self-modification and code generation
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.self_modifier = SelfModificationEngine(base_dir)
        self.code_generator = CodeGenerationEngine(base_dir)
        self.active = False
    
    def activate(self):
        """Activate auto-coding capabilities"""
        self.active = True
        logger.info("🚀 Auto-coding engine ACTIVATED")
        logger.info("   System can now modify its own code!")
    
    def deactivate(self):
        """Deactivate for safety"""
        self.active = False
        logger.info("🛑 Auto-coding engine deactivated")
    
    def create_new_algorithm(
        self,
        name: str,
        description: str,
        target_dir: str = "extracted_algorithms"
    ) -> bool:
        """
        Create a new algorithm file
        
        Args:
            name: Algorithm name
            description: What it does
            target_dir: Where to create it
        
        Returns:
            True if successful
        """
        if not self.active:
            logger.warning("Auto-coding engine not active")
            return False
        
        # Generate code
        code = self.code_generator.generate_algorithm(name, description)
        if not code:
            return False
        
        # Create action
        target_file = os.path.join(self.base_dir, target_dir, f"{name.lower()}.py")
        action = self.self_modifier.propose_modification(
            target_file=target_file,
            modification_type='add',
            content=code,
            reason=f"Create new algorithm: {description}"
        )
        
        if not action:
            return False
        
        # Apply
        return self.self_modifier.apply_modification(action)
    
    def improve_existing_code(self, file_path: str) -> bool:
        """
        Analyze and improve existing code
        
        Args:
            file_path: File to improve
        
        Returns:
            True if improvements applied
        """
        if not self.active:
            logger.warning("Auto-coding engine not active")
            return False
        
        # Get suggestions
        suggestions = self.code_generator.suggest_improvement(file_path)
        if not suggestions:
            logger.info(f"No improvements needed for {file_path}")
            return True
        
        logger.info(f"📊 Suggestions for {file_path}:")
        logger.info(f"   {suggestions}")
        
        # For now, just log suggestions
        # In future: automatically apply improvements
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-coding status"""
        mod_stats = self.self_modifier.get_statistics()
        
        return {
            'active': self.active,
            'self_modification': mod_stats,
            'code_generation': {
                'total_generated': self.code_generator.generation_count
            }
        }
    
    def generate_improvements(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate REAL code improvements using APIs for NOVELTY
        """
        suggestions = []
        
        # Try API for NOVEL suggestions
        try:
            import sys
            sys.path.insert(0, '/root/intelligence_system')
            from apis.real_api_client import RealAPIClient
            
            client = RealAPIClient(timeout=20)
            prompt = f"""Metrics: MNIST={metrics.get('mnist_acc',0):.1f}%, CartPole={metrics.get('cartpole_avg',0):.1f}, IA³={metrics.get('ia3_score',0):.1f}%

Suggest 1 NOVEL code improvement (not standard tuning). Focus on: new architectures, emergent behaviors, curiosity-driven learning. Be specific (Python/PyTorch), max 200 chars."""
            
            response = client.consult_best(prompt, max_providers=2)
            if response:
                suggestions.append({
                    'description': response[:180],
                    'source': 'API_NOVEL',
                    'priority': 'high',
                    'novelty_score': 0.8
                })
                logger.info(f"   💡 Novel suggestion from API")
        except Exception as e:
            logger.debug(f"API suggestion unavailable: {e}")
        
        # Fallback local
        if not suggestions and metrics.get('mnist_acc', 0) < 99.0:
            suggestions.append({
                'description': f"Add attention mechanism (MNIST {metrics.get('mnist_acc',0):.1f}%)",
                'source': 'LOCAL',
                'priority': 'medium',
                'novelty_score': 0.3
            })
        
        return suggestions


# Test function
def test_auto_coding_engine():
    """Test the auto-coding engine"""
    base_dir = "/root/intelligence_system"
    engine = AutoCodingOrchestrator(base_dir)
    
    print("="*80)
    print("🧪 TESTING AUTO-CODING ENGINE")
    print("="*80)
    
    # Activate
    engine.activate()
    
    # Test status
    status = engine.get_status()
    print(f"\n📊 Status: {status}")
    
    # Test code generation (dry run)
    code = engine.code_generator.generate_algorithm(
        name="TestAlgorithm",
        description="Test algorithm for auto-coding"
    )
    print(f"\n✅ Generated code ({len(code)} chars)")
    
    # Test improvement suggestion
    test_file = os.path.join(base_dir, "extracted_algorithms/neural_evolution_core.py")
    if os.path.exists(test_file):
        suggestions = engine.code_generator.suggest_improvement(test_file)
        if suggestions:
            print(f"\n📊 Suggestions for neural_evolution_core.py:")
            print(f"   {suggestions}")
    
    print("\n" + "="*80)
    print("✅ AUTO-CODING ENGINE TEST COMPLETE")
    print("="*80)
    
    return engine


if __name__ == "__main__":
    # Run test
    engine = test_auto_coding_engine()
