#!/usr/bin/env python3
"""
Export API health dashboard as HTML/CSV.
Reads data/api_health.json and generates human-readable reports.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_DIR


def export_html():
    """Generate HTML health dashboard"""
    health_path = Path(DATA_DIR) / 'api_health.json'
    if not health_path.exists():
        print("⚠️ No health data yet")
        return
    
    with open(health_path) as f:
        health = json.load(f)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>API Health Dashboard</title>
    <style>
        body {{ font-family: monospace; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .provider {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .metric {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>🏥 API Health Dashboard</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    for provider, data in health.items():
        success = data.get('success_count', 0)
        errors = data.get('error_count', 0)
        total = success + errors
        rate = (success / total * 100) if total > 0 else 0
        
        status_class = 'success' if rate > 80 else 'error'
        
        html += f"""
    <div class="provider">
        <h2>{provider.upper()}</h2>
        <div class="metric">Success Rate: <span class="{status_class}">{rate:.1f}%</span> ({success}/{total})</div>
        <div class="metric">Avg Latency: {data.get('avg_latency_ms', 0):.0f}ms</div>
        <div class="metric">Last Error: {data.get('last_error', 'None')[:100]}</div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    out_path = Path(DATA_DIR) / 'exports' / 'api_health.html'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"✅ HTML exported: {out_path}")


def export_csv():
    """Generate CSV health summary"""
    health_path = Path(DATA_DIR) / 'api_health.json'
    if not health_path.exists():
        print("⚠️ No health data yet")
        return
    
    with open(health_path) as f:
        health = json.load(f)
    
    csv_lines = ["provider,success_count,error_count,success_rate,avg_latency_ms,last_error"]
    
    for provider, data in health.items():
        success = data.get('success_count', 0)
        errors = data.get('error_count', 0)
        total = success + errors
        rate = (success / total * 100) if total > 0 else 0
        latency = data.get('avg_latency_ms', 0)
        last_error = str(data.get('last_error', ''))[:50].replace(',', ';')
        
        csv_lines.append(f"{provider},{success},{errors},{rate:.2f},{latency:.0f},{last_error}")
    
    out_path = Path(DATA_DIR) / 'exports' / 'api_health.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(csv_lines))
    print(f"✅ CSV exported: {out_path}")


if __name__ == '__main__':
    export_html()
    export_csv()
    print("\n📊 Health dashboard exported!")