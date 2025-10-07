"""
AUTO-REPAIR ACTIVATOR
Ativa auto-repair no sistema rodando sem interrompê-lo
"""
import sys
import time
import json
from pathlib import Path

# Adiciona auto-repair ao path
sys.path.insert(0, '/root/intelligence_system')

from extracted_algorithms.auto_repair.integration_hook import initialize_global_hook

def activate_auto_repair(dry_run: bool = True):
    """
    Ativa auto-repair no sistema atual
    
    Esta função pode ser chamada de dentro do loop rodando
    """
    print("="*80)
    print("🔧 ACTIVATING AUTO-REPAIR SYSTEM")
    print("="*80)
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE PATCHING'}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Inicializa hook global
        hook = initialize_global_hook(dry_run=dry_run)
        
        print("✅ Hook initialized")
        print(f"   Status: {hook.get_status()}")
        print()
        
        # Cria arquivo de status
        status_file = Path("/root/intelligence_system/auto_repair_status.json")
        status = {
            'active': True,
            'dry_run': dry_run,
            'activated_at': time.time(),
            'hook_status': hook.get_status()
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"✅ Status file created: {status_file}")
        print()
        
        # Cria trigger para hot-reload no sistema
        trigger_file = Path("/root/intelligence_system/.auto_repair_trigger")
        with open(trigger_file, 'w') as f:
            f.write(f"AUTO_REPAIR_ACTIVE={time.time()}\n")
            f.write(f"DRY_RUN={dry_run}\n")
        
        print(f"✅ Trigger created: {trigger_file}")
        print()
        
        print("="*80)
        print("🎯 AUTO-REPAIR ACTIVATION COMPLETE")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Sistema detectará trigger no próximo ciclo")
        print("2. Auto-repair será integrado automaticamente")
        print("3. Erros serão capturados e reparados")
        print()
        print(f"Monitor logs: tail -f /root/intelligence_system/auto_repair_agent_comm.log")
        print(f"View results: ls /root/intelligence_system/auto_repair_dryrun_results/")
        print(f"View emergences: ls /root/intelligence_system/auto_repair_evidence/")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Activation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_auto_repair_status():
    """Verifica status do auto-repair"""
    status_file = Path("/root/intelligence_system/auto_repair_status.json")
    
    if not status_file.exists():
        print("❌ Auto-repair not activated")
        return None
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    print("="*80)
    print("📊 AUTO-REPAIR STATUS")
    print("="*80)
    print(json.dumps(status, indent=2))
    print("="*80)
    
    return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-Repair Activator')
    parser.add_argument('--activate', action='store_true', help='Activate auto-repair')
    parser.add_argument('--status', action='store_true', help='Check status')
    parser.add_argument('--live', action='store_true', help='Enable live patching (not dry-run)')
    
    args = parser.parse_args()
    
    if args.status:
        check_auto_repair_status()
    elif args.activate:
        dry_run = not args.live
        activate_auto_repair(dry_run=dry_run)
    else:
        # Default: ativa em dry-run
        activate_auto_repair(dry_run=True)
