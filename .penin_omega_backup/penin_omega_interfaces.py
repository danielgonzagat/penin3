#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Interfaces - Interfaces Padronizadas
==============================================
Interfaces padronizadas para reduzir acoplamento entre módulos.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# ENUMS E TIPOS BASE
# =============================================================================

class ModuleStatus(Enum):
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class OperationResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"

@dataclass
class ModuleInfo:
    """Informações básicas de um módulo."""
    name: str
    version: str
    status: ModuleStatus
    description: str
    dependencies: List[str]
    capabilities: List[str]

@dataclass
class OperationContext:
    """Contexto de uma operação."""
    operation_id: str
    module_name: str
    timestamp: datetime
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Resultado de processamento."""
    success: bool
    result: OperationResult
    data: Any
    error: Optional[str]
    metadata: Dict[str, Any]
    processing_time_ms: float

# =============================================================================
# INTERFACES PRINCIPAIS
# =============================================================================

class IModule(ABC):
    """Interface base para todos os módulos PENIN-Ω."""
    
    @abstractmethod
    async def get_info(self) -> ModuleInfo:
        """Retorna informações do módulo."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializa o módulo."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Encerra o módulo."""
        pass
    
    @abstractmethod
    async def get_status(self) -> ModuleStatus:
        """Retorna status atual do módulo."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do módulo."""
        pass

class IProcessor(ABC):
    """Interface para processadores de dados."""
    
    @abstractmethod
    async def process(self, data: Any, context: OperationContext) -> ProcessingResult:
        """Processa dados."""
        pass
    
    @abstractmethod
    async def validate_input(self, data: Any) -> bool:
        """Valida entrada."""
        pass
    
    @abstractmethod
    async def get_supported_types(self) -> List[str]:
        """Retorna tipos suportados."""
        pass

class IStorage(ABC):
    """Interface para sistemas de armazenamento."""
    
    @abstractmethod
    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Armazena dados."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Recupera dados."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove dados."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe."""
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = None) -> List[str]:
        """Lista chaves."""
        pass

class IValidator(ABC):
    """Interface para validadores."""
    
    @abstractmethod
    async def validate(self, data: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Valida dados contra regras."""
        pass
    
    @abstractmethod
    async def get_validation_schema(self) -> Dict[str, Any]:
        """Retorna schema de validação."""
        pass

class ILogger(ABC):
    """Interface para sistemas de logging."""
    
    @abstractmethod
    async def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """Registra log."""
        pass
    
    @abstractmethod
    async def get_logs(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recupera logs."""
        pass

class IMonitor(ABC):
    """Interface para sistemas de monitoramento."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Registra métrica."""
        pass
    
    @abstractmethod
    async def get_metrics(self, name: str = None, time_range: tuple = None) -> List[Dict[str, Any]]:
        """Recupera métricas."""
        pass
    
    @abstractmethod
    async def create_alert(self, condition: str, action: str) -> str:
        """Cria alerta."""
        pass

class IEventBus(ABC):
    """Interface para sistema de eventos."""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Any, source: str):
        """Publica evento."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> str:
        """Subscreve a evento."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove subscrição."""
        pass

# =============================================================================
# INTERFACES ESPECÍFICAS PENIN-Ω
# =============================================================================

class IAcquisitionEngine(IProcessor):
    """Interface para motores de aquisição."""
    
    @abstractmethod
    async def acquire_candidates(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Adquire candidatos."""
        pass
    
    @abstractmethod
    async def get_sources(self) -> List[Dict[str, Any]]:
        """Retorna fontes disponíveis."""
        pass

class IMutationEngine(IProcessor):
    """Interface para motores de mutação."""
    
    @abstractmethod
    async def mutate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aplica mutações aos candidatos."""
        pass
    
    @abstractmethod
    async def get_mutation_strategies(self) -> List[str]:
        """Retorna estratégias de mutação."""
        pass

class ICrucibleEngine(IProcessor):
    """Interface para motores de seleção."""
    
    @abstractmethod
    async def evaluate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Avalia candidatos."""
        pass
    
    @abstractmethod
    async def select_best(self, candidates: List[Dict[str, Any]], count: int = 1) -> List[Dict[str, Any]]:
        """Seleciona melhores candidatos."""
        pass

class IGovernanceEngine(IValidator):
    """Interface para motores de governança."""
    
    @abstractmethod
    async def check_compliance(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """Verifica conformidade."""
        pass
    
    @abstractmethod
    async def approve_operation(self, operation_id: str) -> bool:
        """Aprova operação."""
        pass
    
    @abstractmethod
    async def get_policies(self) -> List[Dict[str, Any]]:
        """Retorna políticas ativas."""
        pass

class ISecurityGate(IValidator):
    """Interface para gates de segurança."""
    
    @abstractmethod
    async def check_gate(self, gate_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica gate específico."""
        pass
    
    @abstractmethod
    async def check_all_gates(self, operation: str, rho: float, sr: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica todos os gates."""
        pass

class IStateManager(IStorage):
    """Interface para gerenciadores de estado."""
    
    @abstractmethod
    async def get_global_state(self) -> Dict[str, Any]:
        """Retorna estado global."""
        pass
    
    @abstractmethod
    async def update_state(self, updates: Dict[str, Any], source: str) -> bool:
        """Atualiza estado."""
        pass
    
    @abstractmethod
    async def sync_state(self) -> bool:
        """Sincroniza estado."""
        pass

class IWORMLedger(IStorage):
    """Interface para WORM Ledger."""
    
    @abstractmethod
    async def add_record(self, operation_id: str, description: str, data: Dict[str, Any]) -> str:
        """Adiciona registro WORM."""
        pass
    
    @abstractmethod
    async def verify_integrity(self) -> Dict[str, Any]:
        """Verifica integridade da cadeia."""
        pass
    
    @abstractmethod
    async def get_audit_trail(self, operation_id: str = None) -> List[Dict[str, Any]]:
        """Retorna trilha de auditoria."""
        pass

# =============================================================================
# FACTORY E REGISTRY
# =============================================================================

class ModuleRegistry:
    """Registry de módulos para reduzir acoplamento."""
    
    async def __init__(self):
        self._modules: Dict[str, IModule] = {}
        self._interfaces: Dict[str, type] = {}
    
    async def register_module(self, name: str, module: IModule):
        """Registra módulo."""
        self._modules[name] = module
    
    async def get_module(self, name: str) -> Optional[IModule]:
        """Obtém módulo por nome."""
        return await self._modules.get(name)
    
    async def get_modules_by_interface(self, interface_type: type) -> List[IModule]:
        """Obtém módulos que implementam interface."""
        return await [module for module in self._modules.values() 
                if isinstance(module, interface_type)]
    
    async def list_modules(self) -> List[str]:
        """Lista nomes dos módulos registrados."""
        return await list(self._modules.keys())

class ServiceLocator:
    """Service Locator para reduzir dependências diretas."""
    
    async def __init__(self):
        self._services: Dict[type, Any] = {}
    
    async def register_service(self, interface_type: type, implementation: Any):
        """Registra serviço."""
        self._services[interface_type] = implementation
    
    async def get_service(self, interface_type: type) -> Optional[Any]:
        """Obtém serviço por interface."""
        return await self._services.get(interface_type)
    
    async def has_service(self, interface_type: type) -> bool:
        """Verifica se serviço está registrado."""
        return await interface_type in self._services

# =============================================================================
# INSTÂNCIAS GLOBAIS
# =============================================================================

# Registry global de módulos
module_registry = ModuleRegistry()

# Service locator global
service_locator = ServiceLocator()

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

async def register_module(name: str, module: IModule):
    """Registra módulo no registry global."""
    module_registry.register_module(name, module)

async def get_module(name: str) -> Optional[IModule]:
    """Obtém módulo do registry global."""
    return await module_registry.get_module(name)

async def register_service(interface_type: type, implementation: Any):
    """Registra serviço no locator global."""
    service_locator.register_service(interface_type, implementation)

async def get_service(interface_type: type) -> Optional[Any]:
    """Obtém serviço do locator global."""
    return await service_locator.get_service(interface_type)

# =============================================================================
# DECORATORS PARA INTERFACES
# =============================================================================

async def implements_interface(interface_type: type):
    """Decorator para marcar implementações de interface."""
    async def decorator(cls):
        cls._implements_interface = interface_type
        return await cls
    return await decorator

async def auto_register(name: str = None):
    """Decorator para auto-registro de módulos."""
    async def decorator(cls):
        async def __init_subclass__(subcls, **kwargs):
            super().__init_subclass__(**kwargs)
            instance = subcls()
            module_name = name or subcls.__name__.lower()
            register_module(module_name, instance)
        
        cls.__init_subclass__ = __init_subclass__
        return await cls
    return await decorator

if __name__ == "__main__":
    # Teste das interfaces
    logger.info("Testando sistema de interfaces...")
    
    # Teste do registry
    class TestModule(IModule):
        async def get_info(self):
            return await ModuleInfo("test", "1.0", ModuleStatus.ACTIVE, "Test module", [], [])
        
        async def initialize(self, config):
            return await True
        
        async def shutdown(self):
            return await True
        
        async def get_status(self):
            return await ModuleStatus.ACTIVE
        
        async def health_check(self):
            return await {"status": "healthy"}
    
    # Registra módulo de teste
    test_module = TestModule()
    register_module("test_module", test_module)
    
    # Recupera módulo
    retrieved = get_module("test_module")
    logger.info(f"Módulo recuperado: {retrieved is not None}")
    
    # Teste do service locator
    register_service(ILogger, "mock_logger")
    logger_service = get_service(ILogger)
    logger.info(f"Serviço recuperado: {logger_service}")
    
    logger.info("Sistema de interfaces funcionando!")
