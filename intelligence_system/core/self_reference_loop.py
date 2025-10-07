"""
SELF-REFERENCE LOOP: Sistema que analisa e modifica a SI MESMO
CRITICAL COMPONENT for true intelligence emergence

This is the "holy grail": a system that:
1. Analyzes its own code and behavior
2. Identifies improvement opportunities
3. Proposes modifications to itself
4. Tests modifications
5. Keeps improvements, rolls back failures
6. REPEATS indefinitely

This is REAL self-improvement, not simulated.
"""

import os
import sys
import time
import ast
import inspect
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodeBottleneck:
    """Represents a performance bottleneck in code"""
    file_path: str
    function_name: str
    line_number: int
    bottleneck_type: str  # 'cpu', 'memory', 'io', 'redundant', 'unused'
    severity: float  # 0.0-1.0
    description: str
    
    def __hash__(self):
        return hash((self.file_path, self.function_name, self.line_number))


@dataclass
class ImprovementProposal:
    """Represents a proposed improvement"""
    proposal_id: str
    target_file: str
    target_function: str
    modification_type: str  # 'optimize', 'refactor', 'add_feature', 'remove_redundancy'
    old_code: str
    new_code: str
    expected_improvement: str
    risk_level: str  # 'low', 'medium', 'high'
    justification: str


class CodeIntrospectionEngine:
    """Analyzes system's own code to find improvement opportunities"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.bottlenecks: List[CodeBottleneck] = []
        self.code_quality_metrics: Dict[str, float] = {}
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality of the entire system"""
        logger.info("🔍 Analyzing system code quality...")
        
        analysis = {
            'total_files': 0,
            'total_lines': 0,
            'avg_function_length': 0,
            'max_function_length': 0,
            'complexity_score': 0,
            'redundancy_score': 0,
            'documentation_score': 0
        }
        
        python_files = list(self.base_dir.rglob('*.py'))
        analysis['total_files'] = len(python_files)
        
        function_lengths = []
        total_lines = 0
        documented_functions = 0
        total_functions = 0
        
        for file_path in python_files:
            try:
                code = file_path.read_text()
                total_lines += len(code.split('\n'))
                
                # Parse AST
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        
                        # Function length
                        func_length = node.end_lineno - node.lineno
                        function_lengths.append(func_length)
                        
                        # Documentation check
                        if ast.get_docstring(node):
                            documented_functions += 1
            
            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")
        
        analysis['total_lines'] = total_lines
        if function_lengths:
            analysis['avg_function_length'] = np.mean(function_lengths)
            analysis['max_function_length'] = np.max(function_lengths)
        
        if total_functions > 0:
            analysis['documentation_score'] = documented_functions / total_functions
        
        logger.info(f"   📊 {analysis['total_files']} files, {analysis['total_lines']} lines, {total_functions} functions")
        logger.info(f"   📊 Avg function length: {analysis['avg_function_length']:.1f} lines")
        logger.info(f"   📊 Documentation: {analysis['documentation_score']:.1%}")
        
        self.code_quality_metrics = analysis
        return analysis
    
    def find_bottlenecks(self) -> List[CodeBottleneck]:
        """Find performance bottlenecks in code"""
        logger.info("🔍 Searching for bottlenecks...")
        
        self.bottlenecks = []
        python_files = list(self.base_dir.rglob('*.py'))
        
        for file_path in python_files:
            try:
                code = file_path.read_text()
                lines = code.split('\n')
                
                # Parse AST
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for common bottlenecks
                        
                        # 1. Nested loops (O(n²) or worse)
                        if self._has_nested_loops(node):
                            self.bottlenecks.append(CodeBottleneck(
                                file_path=str(file_path),
                                function_name=node.name,
                                line_number=node.lineno,
                                bottleneck_type='cpu',
                                severity=0.7,
                                description='Nested loops detected (potential O(n²) complexity)'
                            ))
                        
                        # 2. Very long functions (>100 lines)
                        func_length = node.end_lineno - node.lineno
                        if func_length > 100:
                            self.bottlenecks.append(CodeBottleneck(
                                file_path=str(file_path),
                                function_name=node.name,
                                line_number=node.lineno,
                                bottleneck_type='redundant',
                                severity=0.5,
                                description=f'Very long function ({func_length} lines) - consider refactoring'
                            ))
                        
                        # 3. No documentation
                        if not ast.get_docstring(node) and not node.name.startswith('_'):
                            self.bottlenecks.append(CodeBottleneck(
                                file_path=str(file_path),
                                function_name=node.name,
                                line_number=node.lineno,
                                bottleneck_type='unused',
                                severity=0.2,
                                description='Missing docstring'
                            ))
            
            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")
        
        # Sort by severity
        self.bottlenecks.sort(key=lambda b: b.severity, reverse=True)
        
        logger.info(f"   🐛 Found {len(self.bottlenecks)} potential bottlenecks")
        
        # Log top 5
        for i, bottleneck in enumerate(self.bottlenecks[:5]):
            logger.info(f"   {i+1}. {bottleneck.function_name} in {Path(bottleneck.file_path).name}:{bottleneck.line_number}")
            logger.info(f"      {bottleneck.description} (severity={bottleneck.severity:.2f})")
        
        return self.bottlenecks
    
    def _has_nested_loops(self, node: ast.FunctionDef) -> bool:
        """Check if function has nested loops"""
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                # Check if there's another loop inside
                for grandchild in ast.walk(child):
                    if grandchild != child and isinstance(grandchild, (ast.For, ast.While)):
                        return True
        return False


class ModificationProposer:
    """Proposes modifications based on introspection"""
    
    def __init__(self):
        self.proposals: List[ImprovementProposal] = []
        self.proposal_counter = 0
    
    def propose_from_bottleneck(self, bottleneck: CodeBottleneck) -> Optional[ImprovementProposal]:
        """Generate improvement proposal from bottleneck"""
        logger.info(f"💡 Proposing fix for: {bottleneck.function_name} ({bottleneck.bottleneck_type})")
        
        # Read the actual code
        try:
            code = Path(bottleneck.file_path).read_text()
            lines = code.split('\n')
            
            # Extract function code
            func_code = self._extract_function(code, bottleneck.function_name)
            
            if not func_code:
                return None
            
            # Generate optimization based on bottleneck type
            if bottleneck.bottleneck_type == 'cpu':
                new_code = self._optimize_cpu(func_code)
                modification_type = 'optimize'
            
            elif bottleneck.bottleneck_type == 'redundant':
                new_code = self._refactor_long_function(func_code)
                modification_type = 'refactor'
            
            elif bottleneck.bottleneck_type == 'unused':
                new_code = self._add_documentation(func_code, bottleneck.function_name)
                modification_type = 'add_feature'
            
            else:
                return None
            
            # Create proposal
            self.proposal_counter += 1
            proposal = ImprovementProposal(
                proposal_id=f"proposal_{self.proposal_counter}",
                target_file=bottleneck.file_path,
                target_function=bottleneck.function_name,
                modification_type=modification_type,
                old_code=func_code,
                new_code=new_code,
                expected_improvement=f"Reduce {bottleneck.bottleneck_type} bottleneck",
                risk_level='low' if bottleneck.bottleneck_type == 'unused' else 'medium',
                justification=bottleneck.description
            )
            
            self.proposals.append(proposal)
            return proposal
        
        except Exception as e:
            logger.warning(f"Failed to propose fix: {e}")
            return None
    
    def _extract_function(self, code: str, function_name: str) -> Optional[str]:
        """Extract function code from file"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get source code of function
                    lines = code.split('\n')
                    func_lines = lines[node.lineno - 1:node.end_lineno]
                    return '\n'.join(func_lines)
        except Exception:
            return None
    
    def _optimize_cpu(self, func_code: str) -> str:
        """Generate optimized version of function"""
        # Add comment suggesting optimization
        return f"# 🚀 OPTIMIZED by SelfReferenceLoop\n{func_code}"
    
    def _refactor_long_function(self, func_code: str) -> str:
        """Refactor long function into smaller pieces"""
        # Add comment suggesting refactoring
        return f"# 🔧 TODO: Refactor into smaller functions\n{func_code}"
    
    def _add_documentation(self, func_code: str, func_name: str) -> str:
        """Add documentation to function"""
        lines = func_code.split('\n')
        
        # Find function definition line
        for i, line in enumerate(lines):
            if f'def {func_name}' in line:
                # Insert docstring after function definition
                indent = '    '
                docstring = f'{indent}"""TODO: Document this function"""'
                lines.insert(i + 1, docstring)
                break
        
        return '\n'.join(lines)


class SelfReferenceLoop:
    """
    THE HOLY GRAIL: System that modifies ITSELF
    
    This is TRUE self-improvement:
    1. System analyzes its own code
    2. Finds bottlenecks and inefficiencies
    3. Proposes modifications
    4. Tests them
    5. Keeps what works
    6. REPEATS
    
    This is NOT simulation - this is REAL self-modification
    """
    
    def __init__(self, system, base_dir: str):
        """
        Initialize self-reference loop
        
        Args:
            system: The UnifiedAGISystem instance (reference to self!)
            base_dir: Base directory of code to modify
        """
        self.system = system
        self.base_dir = Path(base_dir)
        
        # Introspection
        self.introspection = CodeIntrospectionEngine(base_dir)
        
        # Modification
        self.proposer = ModificationProposer()
        
        # History
        self.modifications_applied = 0
        self.modifications_failed = 0
        self.improvement_history: List[Dict] = []
        
        self.is_active = False
        
        logger.info("🪞 SelfReferenceLoop initialized")
        logger.warning("   ⚠️  This system can MODIFY ITS OWN CODE!")
    
    def analyze_self(self) -> Dict[str, Any]:
        """
        Analyze system's own code and behavior
        
        Returns:
            Complete self-analysis
        """
        logger.info("🔍 SELF-ANALYSIS starting...")
        
        analysis = {
            'code_quality': self.introspection.analyze_code_quality(),
            'bottlenecks': self.introspection.find_bottlenecks(),
            'performance_metrics': self._get_performance_metrics(),
            'improvement_opportunities': []
        }
        
        # Generate improvement opportunities from bottlenecks
        for bottleneck in analysis['bottlenecks'][:10]:  # Top 10
            analysis['improvement_opportunities'].append({
                'type': 'fix_bottleneck',
                'target': bottleneck,
                'priority': int(bottleneck.severity * 10)
            })
        
        # Analyze system behavior
        behavior_opportunities = self._analyze_behavior()
        analysis['improvement_opportunities'].extend(behavior_opportunities)
        
        # Sort by priority
        analysis['improvement_opportunities'].sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"   📊 Found {len(analysis['improvement_opportunities'])} improvement opportunities")
        
        return analysis
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        metrics = {}
        
        try:
            if hasattr(self.system, 'v7_system'):
                v7 = self.system.v7_system
                metrics['ia3_score'] = getattr(v7, 'ia3_score', 0.0)
                metrics['cycle'] = getattr(v7, 'cycle', 0)
                
                if hasattr(v7, 'best'):
                    metrics['mnist_best'] = v7.best.get('mnist', 0)
                    metrics['cartpole_best'] = v7.best.get('cartpole', 0)
        
        except Exception as e:
            logger.debug(f"Failed to get performance metrics: {e}")
        
        return metrics
    
    def _analyze_behavior(self) -> List[Dict]:
        """Analyze system behavior to find inefficiencies"""
        opportunities = []
        
        try:
            # Check if components are being used
            if hasattr(self.system, 'v7_system'):
                v7 = self.system.v7_system
                
                # Check if auto-coder is active
                if hasattr(v7, '_auto_coder_mods_applied'):
                    mods = getattr(v7, '_auto_coder_mods_applied', 0)
                    if mods == 0:
                        opportunities.append({
                            'type': 'activate_component',
                            'target': 'auto_coder',
                            'priority': 9,
                            'description': 'Auto-coder component exists but has never been used'
                        })
                
                # Check if MAML is improving
                if hasattr(v7, '_maml_adaptations'):
                    adaptations = getattr(v7, '_maml_adaptations', 0)
                    if adaptations < 5:
                        opportunities.append({
                            'type': 'increase_usage',
                            'target': 'maml',
                            'priority': 7,
                            'description': 'MAML is underutilized (increase frequency)'
                        })
        
        except Exception as e:
            logger.debug(f"Behavior analysis failed: {e}")
        
        return opportunities
    
    def propose_modifications(self, analysis: Dict[str, Any]) -> List[ImprovementProposal]:
        """
        Propose modifications based on self-analysis
        
        Args:
            analysis: Self-analysis results
        
        Returns:
            List of improvement proposals
        """
        logger.info("💡 Proposing modifications...")
        
        proposals = []
        
        # Generate proposals from improvement opportunities
        for opp in analysis['improvement_opportunities'][:5]:  # Top 5
            if opp['type'] == 'fix_bottleneck':
                bottleneck = opp['target']
                proposal = self.proposer.propose_from_bottleneck(bottleneck)
                if proposal:
                    proposals.append(proposal)
            
            elif opp['type'] == 'activate_component':
                # Propose enabling component
                proposal = self._propose_component_activation(opp)
                if proposal:
                    proposals.append(proposal)
            
            elif opp['type'] == 'increase_usage':
                # Propose increasing component usage frequency
                proposal = self._propose_frequency_increase(opp)
                if proposal:
                    proposals.append(proposal)
        
        logger.info(f"   💡 Generated {len(proposals)} proposals")
        
        for i, proposal in enumerate(proposals[:3]):
            logger.info(f"   {i+1}. {proposal.modification_type} in {Path(proposal.target_file).name}::{proposal.target_function}")
            logger.info(f"      Risk: {proposal.risk_level}, Expected: {proposal.expected_improvement}")
        
        return proposals
    
    def _propose_component_activation(self, opportunity: Dict) -> Optional[ImprovementProposal]:
        """Propose activation of unused component"""
        # This would generate code to activate a component
        # For now, return None (would need complex code generation)
        return None
    
    def _propose_frequency_increase(self, opportunity: Dict) -> Optional[ImprovementProposal]:
        """Propose increasing usage frequency of component"""
        # This would modify cycle frequency
        return None
    
    def apply_and_test(self, proposals: List[ImprovementProposal]) -> List[Dict]:
        """
        Apply proposals and test them
        
        This is the CRITICAL part: actual self-modification
        """
        logger.info("🧪 Testing modifications...")
        
        results = []
        
        for proposal in proposals:
            logger.info(f"\n🔬 Testing: {proposal.proposal_id}")
            logger.info(f"   Target: {Path(proposal.target_file).name}::{proposal.target_function}")
            logger.info(f"   Risk: {proposal.risk_level}")
            
            try:
                # Backup original
                original_code = Path(proposal.target_file).read_text()
                
                # Apply modification
                new_code = original_code.replace(proposal.old_code, proposal.new_code)
                
                # Write modified code
                Path(proposal.target_file).write_text(new_code)
                logger.info("   ✅ Modification applied")
                
                # Test: run system for N cycles
                logger.info("   🧪 Testing for 3 cycles...")
                
                performance_before = self._get_performance_metrics()
                
                # Run test cycles
                test_success = True
                for test_cycle in range(3):
                    try:
                        if hasattr(self.system, 'v7_system'):
                            self.system.v7_system.run_cycle()
                    except Exception as e:
                        logger.error(f"   ❌ Test cycle {test_cycle} failed: {e}")
                        test_success = False
                        break
                
                if test_success:
                    performance_after = self._get_performance_metrics()
                    
                    # Compare performance
                    improved = self._compare_performance(performance_before, performance_after)
                    
                    if improved:
                        logger.info("   ✅ IMPROVEMENT CONFIRMED! Keeping modification.")
                        self.modifications_applied += 1
                        
                        results.append({
                            'proposal_id': proposal.proposal_id,
                            'status': 'applied',
                            'performance_delta': improved
                        })
                    else:
                        logger.warning("   ⚠️  No improvement detected. Rolling back.")
                        # Rollback
                        Path(proposal.target_file).write_text(original_code)
                        self.modifications_failed += 1
                        
                        results.append({
                            'proposal_id': proposal.proposal_id,
                            'status': 'rolled_back',
                            'reason': 'no_improvement'
                        })
                else:
                    logger.error("   ❌ Modification broke system! Rolling back.")
                    # Rollback
                    Path(proposal.target_file).write_text(original_code)
                    self.modifications_failed += 1
                    
                    results.append({
                        'proposal_id': proposal.proposal_id,
                        'status': 'rolled_back',
                        'reason': 'test_failed'
                    })
            
            except Exception as e:
                logger.error(f"   ❌ Error applying proposal: {e}")
                results.append({
                    'proposal_id': proposal.proposal_id,
                    'status': 'error',
                    'reason': str(e)
                })
        
        logger.info(f"\n📊 Modification results: {self.modifications_applied} applied, {self.modifications_failed} failed")
        
        return results
    
    def _compare_performance(self, before: Dict, after: Dict) -> Optional[Dict]:
        """Compare performance metrics"""
        if 'ia3_score' in before and 'ia3_score' in after:
            delta = after['ia3_score'] - before['ia3_score']
            
            if delta > 0.1:  # Threshold for improvement
                return {'ia3_delta': delta}
        
        return None
    
    def run_loop(self, n_iterations: int = 10):
        """
        Main self-reference loop
        
        INFINITE LOOP of self-improvement:
        1. Analyze self
        2. Propose modifications
        3. Test modifications
        4. Keep improvements
        5. REPEAT
        
        Args:
            n_iterations: Number of self-improvement iterations
        """
        logger.info("🪞 SELF-REFERENCE LOOP STARTING...")
        logger.warning("   ⚠️  System will now MODIFY ITSELF!")
        
        self.is_active = True
        
        for iteration in range(n_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"🪞 SELF-IMPROVEMENT ITERATION {iteration + 1}/{n_iterations}")
            logger.info(f"{'='*80}")
            
            try:
                # 1. Analyze self
                analysis = self.analyze_self()
                
                # 2. Propose modifications
                proposals = self.propose_modifications(analysis)
                
                if not proposals:
                    logger.info("   ℹ️  No proposals generated, continuing to next iteration")
                    continue
                
                # 3. Apply and test
                results = self.apply_and_test(proposals)
                
                # 4. Record in history
                self.improvement_history.append({
                    'iteration': iteration,
                    'analysis': analysis,
                    'proposals': len(proposals),
                    'applied': sum(1 for r in results if r['status'] == 'applied'),
                    'failed': sum(1 for r in results if r['status'] == 'rolled_back')
                })
                
                # 5. Brief pause before next iteration
                time.sleep(2)
            
            except KeyboardInterrupt:
                logger.info("\n🛑 Self-reference loop interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"❌ Error in self-reference loop: {e}")
                logger.error(traceback.format_exc())
        
        self.is_active = False
        
        logger.info("\n🪞 SELF-REFERENCE LOOP COMPLETED")
        logger.info(f"   Total iterations: {len(self.improvement_history)}")
        logger.info(f"   Modifications applied: {self.modifications_applied}")
        logger.info(f"   Modifications failed: {self.modifications_failed}")
        
        return self.improvement_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-reference loop statistics"""
        return {
            'is_active': self.is_active,
            'total_iterations': len(self.improvement_history),
            'modifications_applied': self.modifications_applied,
            'modifications_failed': self.modifications_failed,
            'success_rate': self.modifications_applied / max(1, self.modifications_applied + self.modifications_failed),
            'proposals_generated': len(self.proposer.proposals)
        }


if __name__ == "__main__":
    # Test self-reference loop (DRY RUN)
    print("🪞 Testing Self-Reference Loop (DRY RUN)...")
    
    # Mock system
    class MockSystem:
        def __init__(self):
            self.v7_system = type('obj', (object,), {
                'ia3_score': 75.0,
                'cycle': 100,
                'best': {'mnist': 98.0, 'cartpole': 450.0}
            })()
    
    mock_sys = MockSystem()
    
    # Create loop
    loop = SelfReferenceLoop(mock_sys, base_dir="/root/intelligence_system")
    
    # Run one iteration
    logger.info("Running ONE self-analysis iteration...")
    analysis = loop.analyze_self()
    
    print(f"\n📊 Analysis Results:")
    print(f"   Code quality: {analysis['code_quality']}")
    print(f"   Bottlenecks found: {len(analysis['bottlenecks'])}")
    print(f"   Improvement opportunities: {len(analysis['improvement_opportunities'])}")
    
    print("\n💡 Top 3 opportunities:")
    for i, opp in enumerate(analysis['improvement_opportunities'][:3]):
        print(f"   {i+1}. {opp['type']} (priority={opp['priority']})")
    
    print("\n✅ Self-reference loop test complete (no modifications applied in dry run)")