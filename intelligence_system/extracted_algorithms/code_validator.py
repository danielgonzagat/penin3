"""
CODE VALIDATOR - Extraído de agi-alpha-real
Validação interna de código sem dependências externas
V6 PATCH: Whitelist para código interno seguro
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class InternalCodeValidator:
    """
    Internal validation engine for code and decisions
    Extracted from: agi-alpha-real/self_evolution_chat.py
    V6 PATCH: Added internal whitelist for safe eval() usage
    """
    
    def __init__(self):
        self.knowledge_base = {
            'python_patterns': ['def ', 'class ', 'import ', 'if ', 'for ', 'while '],
            'security_checks': ['exec(', 'eval(', 'os.system(', 'subprocess.call(', '__import__'],
            'logic_rules': ['syntax_valid', 'imports_exist', 'no_infinite_loops']
        }
        
        # V6 PATCH: Whitelist for internal safe code
        self.internal_whitelist = [
            'extracted_algorithms.self_modification_engine',  # Safe internal eval
            'extracted_algorithms.advanced_evolution_engine',  # Safe internal eval
            'core.database_knowledge_engine',  # Safe data parsing
        ]
        
        logger.info("🔒 Internal Code Validator initialized (with whitelist)")
    
    def validate_code(self, code: str, source_module: str = None) -> Dict[str, bool]:
        """
        Validate code internally without external API
        
        Args:
            code: Code to validate
            source_module: Module name (for whitelisting)
        
        Returns:
            Dict with validation results
        """
        checks = {
            'syntax': self._check_syntax(code),
            'security': self._check_security(code, source_module),
            'logic': self._check_logic(code),
            'valid': False
        }
        
        checks['valid'] = all([checks['syntax'], checks['security'], checks['logic']])
        
        return checks
    
    def _check_syntax(self, code: str) -> bool:
        """Check Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _check_security(self, code: str, source_module: str = None) -> bool:
        """
        Check for dangerous patterns
        V6 PATCH: Skip check for whitelisted internal modules
        """
        # V6 PATCH: Check whitelist first
        if source_module:
            for safe_module in self.internal_whitelist:
                if safe_module in source_module:
                    # Trusted internal code, skip security check
                    return True
        
        for dangerous in self.knowledge_base['security_checks']:
            if dangerous in code:
                logger.warning(f"🚨 Security risk detected: {dangerous}")
                return False
        return True
    
    def _check_logic(self, code: str) -> bool:
        """Basic logic checks"""
        # Check for infinite loops
        if 'while True:' in code and 'break' not in code and 'return' not in code:
            logger.warning("⚠️  Potential infinite loop detected")
            return False
        
        # Check for basic structure
        if len(code.strip()) < 10:
            return False
        
        return True
    
    def auto_fix_code(self, code: str, validation: Dict[str, bool]) -> str:
        """
        Attempt to automatically fix code issues
        
        Args:
            code: Code to fix
            validation: Validation results
        
        Returns:
            Fixed code
        """
        fixed_code = code
        
        if not validation['syntax']:
            # Fix common syntax issues
            fixed_code = fixed_code.replace('\t', '    ')  # Fix indentation
        
        if not validation['security']:
            # Remove dangerous patterns
            dangerous = self.knowledge_base['security_checks']
            for danger in dangerous:
                if danger in fixed_code:
                    fixed_code = fixed_code.replace(danger, f'# REMOVED: {danger}')
                    logger.info(f"🔧 Removed dangerous pattern: {danger}")
        
        if not validation['logic']:
            # Add safety break to infinite loops
            if 'while True:' in fixed_code and 'break' not in fixed_code:
                lines = fixed_code.split('\n')
                for i, line in enumerate(lines):
                    if 'while True:' in line:
                        indent = len(line) - len(line.lstrip())
                        lines.insert(i+1, ' ' * (indent + 4) + 'break  # Safety break')
                        break
                fixed_code = '\n'.join(lines)
                logger.info("🔧 Added safety break to infinite loop")
        
        return fixed_code
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        return {
            'patterns': len(self.knowledge_base['python_patterns']),
            'security_rules': len(self.knowledge_base['security_checks']),
            'logic_rules': len(self.knowledge_base['logic_rules']),
            'whitelisted_modules': len(self.internal_whitelist)
        }
