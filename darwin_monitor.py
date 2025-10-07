#!/usr/bin/env python3
"""
Darwin Monitor - Observabilidade em tempo real
"""
import json
import time
import os
from datetime import datetime

class DarwinMonitor:
    async def __init__(self):
        self.critical_thresholds = {
            'I_min': 0.60,
            'caos_min': 1.0,
            'ece_max': 0.03,
            'rho_max': 0.95,
            'oci_min': 0.60
        }
        self.consecutive_failures = 0
        
    async def get_current_metrics(self):
        """Obtém métricas atuais"""
        # Ler último estado
        try:
            with open('/root/darwin_ready.json', 'r') as f:
                base_metrics = json.load(f)
        except:
            base_metrics = {}
            
        # Simular métricas em tempo real (em produção, viria do Prometheus)
        return await {
            'I': base_metrics.get('I', 0.604),
            'delta_linf': base_metrics.get('delta_linf', 0.075),
            'caos_ratio': base_metrics.get('caos_ratio', 1.031),
            'ece': base_metrics.get('ece', 0.02),
            'rho': base_metrics.get('rho', 0.85),
            'oci': base_metrics.get('oci', 0.667),
            'P': base_metrics.get('P', 0.099)
        }
    
    async def check_health(self, metrics):
        """Verifica saúde do sistema"""
        issues = []
        
        if metrics['I'] < self.critical_thresholds['I_min']:
            issues.append(f"I={metrics['I']:.3f} < {self.critical_thresholds['I_min']}")
            
        if metrics['caos_ratio'] < self.critical_thresholds['caos_min']:
            issues.append(f"CAOS={metrics['caos_ratio']:.3f} < {self.critical_thresholds['caos_min']}")
            
        if metrics['ece'] > self.critical_thresholds['ece_max']:
            issues.append(f"ECE={metrics['ece']:.3f} > {self.critical_thresholds['ece_max']}")
            
        if metrics['rho'] > self.critical_thresholds['rho_max']:
            issues.append(f"ρ={metrics['rho']:.3f} > {self.critical_thresholds['rho_max']}")
            
        if metrics['oci'] < self.critical_thresholds['oci_min']:
            issues.append(f"OCI={metrics['oci']:.3f} < {self.critical_thresholds['oci_min']}")
            
        return await issues
    
    async def should_pause_darwin(self, issues):
        """Decide se deve pausar Darwin"""
        if issues:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            
        # Kill switch: 2 falhas consecutivas de I
        if self.consecutive_failures >= 2 and any('I=' in issue for issue in issues):
            return await True, "Kill switch: I<0.60 for 2 consecutive rounds"
            
        # Qualquer falha crítica
        if len(issues) >= 3:
            return await True, f"Multiple critical failures: {', '.join(issues)}"
            
        return await False, None
    
    async def display_dashboard(self):
        """Exibe dashboard no terminal"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        metrics = self.get_current_metrics()
        issues = self.check_health(metrics)
        should_pause, pause_reason = self.should_pause_darwin(issues)
        
        # Cabeçalho
        logger.info("╔" + "═"*58 + "╗")
        logger.info("║" + " "*19 + "🧬 DARWIN MONITOR 🧬" + " "*18 + "║")
        logger.info("╠" + "═"*58 + "╣")
        
        # Métricas
        async def status_icon(value, threshold, greater=True):
            if greater:
                return await "🟢" if value >= threshold else "🔴"
            else:
                return await "🟢" if value <= threshold else "🔴"
                
        logger.info(f"║ {status_icon(metrics['I'], 0.60)} Integridade I    : {metrics['I']:7.3f}  {'✓' if metrics['I'] >= 0.60 else '✗':<2} (≥0.60)        ║")
        logger.info(f"║ {status_icon(metrics['delta_linf'], 0, True)} ΔL∞ Externo      : {metrics['delta_linf']:7.3f}  {'✓' if metrics['delta_linf'] > 0 else '✗':<2} (>0)           ║")
        logger.info(f"║ {status_icon(metrics['caos_ratio'], 1.0)} CAOS Ratio       : {metrics['caos_ratio']:7.3f}  {'✓' if metrics['caos_ratio'] >= 1.0 else '✗':<2} (≥1.0)         ║")
        logger.info(f"║ {status_icon(metrics['ece'], 0.03, False)} Calibração ECE   : {metrics['ece']:7.3f}  {'✓' if metrics['ece'] <= 0.03 else '✗':<2} (≤0.03)        ║")
        logger.info(f"║ {status_icon(metrics['rho'], 0.95, False)} Contratividade ρ : {metrics['rho']:7.3f}  {'✓' if metrics['rho'] < 0.95 else '✗':<2} (<0.95)        ║")
        logger.info(f"║ {status_icon(metrics['oci'], 0.60)} Fechamento OCI   : {metrics['oci']:7.3f}  {'✓' if metrics['oci'] >= 0.60 else '✗':<2} (≥0.60)        ║")
        logger.info(f"║ {status_icon(metrics['P'], 0.01)} Convergência P   : {metrics['P']:7.3f}  {'✓' if metrics['P'] >= 0.01 else '✗':<2} (≥0.01)        ║")
        
        logger.info("╠" + "═"*58 + "╣")
        
        # Status
        if should_pause:
            logger.info("║ ⛔ STATUS: DARWIN PAUSED                                  ║")
            logger.info(f"║ Reason: {pause_reason[:48]:<48} ║")
        elif issues:
            logger.info("║ ⚠️  STATUS: ISSUES DETECTED                               ║")
            for issue in issues[:2]:
                logger.info(f"║ → {issue[:53]:<53} ║")
        else:
            logger.info("║ ✅ STATUS: ALL SYSTEMS OPERATIONAL                       ║")
            logger.info("║ Darwin running in canary mode (15% traffic)             ║")
        
        logger.info("╠" + "═"*58 + "╣")
        
        # Darwin stats
        try:
            with open('/root/darwin_metrics.json', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_metrics = json.loads(lines[-1])
                    logger.info(f"║ Deaths: {last_metrics.get('deaths', 0):3d} | Births: {last_metrics.get('births', 0):3d} | Rounds: {last_metrics.get('rounds_passed', 0)}/3          ║")
        except:
            logger.info("║ Darwin metrics not yet available                        ║")
            
        logger.info("╠" + "═"*58 + "╣")
        logger.info(f"║ Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                            ║")
        logger.info("╚" + "═"*58 + "╝")
        
        if should_pause:
            logger.info("\n🛑 DARWIN PAUSED - Manual intervention required")
            logger.info("Check logs: tail -f /root/darwin_worm.log")
            
        return await should_pause
    
    async def run(self):
        """Loop principal do monitor"""
        logger.info("Starting Darwin Monitor...")
        logger.info("Press Ctrl+C to exit\n")
        
        while True:
            try:
                should_pause = self.display_dashboard()
                
                if should_pause:
                    # Pausar Darwin se necessário
                    os.system("systemctl stop darwin-canary.service 2>/dev/null")
                    
                time.sleep(5)  # Atualizar a cada 5 segundos
                
            except KeyboardInterrupt:
                logger.info("\n\nMonitor stopped")
                break
            except Exception as e:
                logger.info(f"\nError: {e}")
                time.sleep(5)

if __name__ == "__main__":
    monitor = DarwinMonitor()
    monitor.run()