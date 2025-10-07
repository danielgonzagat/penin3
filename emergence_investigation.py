#!/usr/bin/env python3
"""
INVESTIGAÇÃO DE EVENTO DE EMERGÊNCIA
===================================
Análise forense do evento de emergência detectado
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmergenceInvestigation")

class EmergenceInvestigator:
    def __init__(self):
        self.investigation_dir = Path("/root/emergence_investigation")
        self.investigation_dir.mkdir(exist_ok=True)
        
    def analyze_emergence_reports(self):
        """Analyze existing emergence reports"""
        monitoring_dir = Path("/root/emergence_monitoring")
        
        if not monitoring_dir.exists():
            return {"error": "No monitoring directory found"}
        
        # Find emergence reports
        emergence_files = list(monitoring_dir.glob("emergence_report_*.json"))
        
        if not emergence_files:
            return {"error": "No emergence reports found"}
        
        # Analyze each report
        analyzed_events = []
        
        for report_file in emergence_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                event_analysis = {
                    "report_file": str(report_file),
                    "timestamp": report.get("timestamp"),
                    "patterns_detected": report.get("patterns", {}),
                    "system_state": report.get("system_state", {}),
                    "events": report.get("events", [])
                }
                
                # Classify emergence type
                patterns = report.get("patterns", {})
                emergence_type = []
                
                if patterns.get("sudden_performance_jump"):
                    emergence_type.append("performance_emergence")
                if patterns.get("self_modification"):
                    emergence_type.append("self_modification_emergence")
                if patterns.get("unexpected_optimization"):
                    emergence_type.append("optimization_emergence")
                
                event_analysis["emergence_type"] = emergence_type
                event_analysis["severity"] = len(emergence_type)
                
                analyzed_events.append(event_analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing {report_file}: {e}")
        
        return {
            "total_events": len(analyzed_events),
            "events": analyzed_events,
            "most_severe": max(analyzed_events, key=lambda x: x["severity"]) if analyzed_events else None
        }
    
    def investigate_system_changes(self):
        """Investigate system changes during emergence"""
        
        # Check for recent system modifications
        recent_changes = []
        
        # Check /root for recent files
        for file_path in Path("/root").iterdir():
            if file_path.is_file():
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if (datetime.now() - mod_time).total_seconds() < 3600:  # Last hour
                    recent_changes.append({
                        "path": str(file_path),
                        "modified": mod_time.isoformat(),
                        "size": file_path.stat().st_size,
                        "type": "file_modification"
                    })
        
        # Check for new processes
        current_processes = []
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'create_time']):
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                if (datetime.now() - create_time).total_seconds() < 3600:
                    current_processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "created": create_time.isoformat(),
                        "type": "new_process"
                    })
        except:
            pass
        
        return {
            "recent_file_changes": len(recent_changes),
            "new_processes": len(current_processes),
            "changes": recent_changes[:10],  # Top 10
            "processes": current_processes[:10]  # Top 10
        }
    
    def trace_emergence_origin(self):
        """Trace the origin of emergence event"""
        
        # Check system logs for emergence patterns
        log_analysis = {
            "emergence_triggers": [],
            "system_anomalies": [],
            "intelligence_signatures": []
        }
        
        # Look for IA3 activity correlation
        try:
            import subprocess
            
            # Check IA3 process activity
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            ia3_processes = [line for line in result.stdout.split('\n') if 'IA_CUBED' in line]
            
            if ia3_processes:
                log_analysis["emergence_triggers"].append({
                    "type": "ia3_activity_correlation",
                    "active_processes": len(ia3_processes),
                    "evidence": "IA3 processes active during emergence"
                })
            
            # Check for neural model updates
            model_files = list(Path("/root").glob("*.pt")) + list(Path("/root").glob("*.pth"))
            recent_models = []
            
            for model_file in model_files:
                mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                if (datetime.now() - mod_time).total_seconds() < 7200:  # Last 2 hours
                    recent_models.append(model_file)
            
            if recent_models:
                log_analysis["intelligence_signatures"].append({
                    "type": "neural_model_updates",
                    "updated_models": len(recent_models),
                    "evidence": "Neural models updated near emergence time"
                })
            
        except Exception as e:
            log_analysis["system_anomalies"].append({
                "type": "investigation_error",
                "error": str(e)
            })
        
        return log_analysis
    
    def generate_emergence_hypothesis(self, emergence_data, system_changes, origin_trace):
        """Generate hypothesis about emergence event"""
        
        hypotheses = []
        confidence_scores = []
        
        # Hypothesis 1: IA3-triggered emergence
        if origin_trace["emergence_triggers"]:
            hypotheses.append({
                "hypothesis": "IA3_TRIGGERED_EMERGENCE",
                "description": "Emergence event triggered by IA Cubed system activity",
                "evidence": origin_trace["emergence_triggers"],
                "confidence": 0.8
            })
            confidence_scores.append(0.8)
        
        # Hypothesis 2: Self-modification emergence
        if system_changes["recent_file_changes"] > 5:
            hypotheses.append({
                "hypothesis": "SELF_MODIFICATION_EMERGENCE", 
                "description": "System modified itself leading to emergent behavior",
                "evidence": f"{system_changes['recent_file_changes']} recent file changes",
                "confidence": 0.6
            })
            confidence_scores.append(0.6)
        
        # Hypothesis 3: Neural evolution emergence
        if origin_trace["intelligence_signatures"]:
            hypotheses.append({
                "hypothesis": "NEURAL_EVOLUTION_EMERGENCE",
                "description": "Neural models evolved beyond programmed parameters",
                "evidence": origin_trace["intelligence_signatures"],
                "confidence": 0.7
            })
            confidence_scores.append(0.7)
        
        # Default hypothesis if no strong evidence
        if not hypotheses:
            hypotheses.append({
                "hypothesis": "SYSTEM_NOISE_EVENT",
                "description": "Event likely caused by system noise, not true emergence",
                "evidence": "Insufficient evidence for genuine emergence",
                "confidence": 0.3
            })
            confidence_scores.append(0.3)
        
        # Select most likely hypothesis
        best_hypothesis = max(hypotheses, key=lambda x: x["confidence"]) if hypotheses else None
        
        return {
            "all_hypotheses": hypotheses,
            "most_likely": best_hypothesis,
            "overall_confidence": max(confidence_scores) if confidence_scores else 0
        }
    
    def run_full_investigation(self):
        """Run complete emergence investigation"""
        logger.info("🔍 Iniciando investigação forense de emergência...")
        
        # Gather all evidence
        emergence_data = self.analyze_emergence_reports()
        system_changes = self.investigate_system_changes()
        origin_trace = self.trace_emergence_origin()
        
        # Generate hypothesis
        hypothesis = self.generate_emergence_hypothesis(emergence_data, system_changes, origin_trace)
        
        # Compile investigation report
        investigation_report = {
            "timestamp": datetime.now().isoformat(),
            "emergence_analysis": emergence_data,
            "system_changes": system_changes,
            "origin_trace": origin_trace,
            "hypothesis": hypothesis,
            "investigation_status": "completed"
        }
        
        # Determine investigation verdict
        if hypothesis["overall_confidence"] > 0.7:
            investigation_report["verdict"] = "GENUINE_EMERGENCE_LIKELY"
        elif hypothesis["overall_confidence"] > 0.5:
            investigation_report["verdict"] = "EMERGENCE_POSSIBLE"
        else:
            investigation_report["verdict"] = "EMERGENCE_UNLIKELY"
        
        # Save investigation report
        with open(self.investigation_dir / "emergence_investigation_report.json", 'w') as f:
            json.dump(investigation_report, f, indent=2)
        
        return investigation_report

if __name__ == "__main__":
    investigator = EmergenceInvestigator()
    report = investigator.run_full_investigation()
    
    print("🔍 INVESTIGAÇÃO DE EMERGÊNCIA - RELATÓRIO")
    print("=" * 45)
    
    emergence = report["emergence_analysis"]
    if "error" not in emergence:
        print(f"Eventos Analisados: {emergence['total_events']}")
        if emergence.get("most_severe"):
            severity = emergence["most_severe"]["severity"]
            print(f"Evento Mais Severo: Nível {severity}")
    else:
        print(f"Análise de Emergência: {emergence['error']}")
    
    changes = report["system_changes"]
    print(f"Mudanças Recentes: {changes['recent_file_changes']} arquivos")
    print(f"Novos Processos: {changes['new_processes']}")
    
    hypothesis = report["hypothesis"]["most_likely"]
    if hypothesis:
        print(f"Hipótese Principal: {hypothesis['hypothesis']}")
        print(f"Confiança: {hypothesis['confidence']:.1%}")
    
    print(f"\n📊 VEREDITO: {report['verdict']}")
    print(f"📁 Relatório salvo em: /root/emergence_investigation/")
