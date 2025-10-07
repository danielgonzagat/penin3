# PENIN-Ω - Guia de Migração

## Compatibilidade Mantida
- Todos os imports existentes funcionam
- APIs públicas preservadas
- Estrutura de dados compatível

## Novos Caminhos
- Módulos: `~/.penin_omega/modules/`
- Logs: `~/.penin_omega/logs/`
- Config: `~/.penin_omega/config/`
- WORM: `~/.penin_omega/worm/`

## Uso Unificado
```python
from penin_omega_master_system import penin_omega
status = penin_omega.get_life_status()
result = penin_omega.execute_full_pipeline("query")
```
