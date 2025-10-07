#!/usr/bin/env python3
"""
✅ IA³ Signal Calculator - ATIVAÇÃO AUTÔNOMA
Calcula e salva IA³ signal no database continuamente
"""

import time
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = "/root/intelligence_system/data/intelligence.db"

def calculate_ia3_from_metrics(episode_data: dict) -> float:
    """
    Calcula IA³ = f(adaptação, desempenho, curiosidade)
    """
    # Componentes do IA³
    coherence = episode_data.get('coherence', 0.0)
    novelty = episode_data.get('novelty', 0.0)
    num_active = episode_data.get('num_active_neurons', 0)
    
    # Score de adaptação (baseado em coherence alta e constante)
    adapt_score = min(1.0, coherence)
    
    # Score de exploração (baseado em novelty)
    explore_score = min(1.0, novelty * 10.0)  # Novelty ~0.04-0.15
    
    # Score de uso efetivo (neurônios ativos)
    usage_score = min(1.0, num_active / 10.0)
    
    # IA³ = média ponderada
    ia3 = (
        adapt_score * 0.5 +
        explore_score * 0.3 +
        usage_score * 0.2
    )
    
    return float(ia3)

def update_ia3_signals():
    """Atualiza IA³ signals de episódios que não têm"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Buscar episódios sem IA³ signal (ou com 0.0)
    cur.execute("""
        SELECT id, episode, coherence, novelty, num_active_neurons
        FROM brain_metrics
        WHERE ia3_signal IS NULL OR ia3_signal = 0.0
        ORDER BY timestamp DESC
        LIMIT 100
    """)
    
    rows = cur.fetchall()
    
    if not rows:
        print("ℹ️  Todos episódios já têm IA³ signal")
        return 0
    
    updated = 0
    for row_id, episode, coherence, novelty, num_active in rows:
        episode_data = {
            'coherence': coherence or 0.0,
            'novelty': novelty or 0.0,
            'num_active_neurons': num_active or 0
        }
        
        ia3 = calculate_ia3_from_metrics(episode_data)
        
        cur.execute("""
            UPDATE brain_metrics
            SET ia3_signal = ?
            WHERE id = ?
        """, (ia3, row_id))
        
        updated += 1
    
    conn.commit()
    conn.close()
    
    return updated

if __name__ == "__main__":
    print(f"🔄 IA³ Calculator starting at {datetime.now()}")
    
    cycle = 0
    while True:
        try:
            cycle += 1
            updated = update_ia3_signals()
            
            if updated > 0:
                print(f"✅ Cycle {cycle}: Updated {updated} IA³ signals")
            else:
                print(f"ℹ️  Cycle {cycle}: No updates needed")
            
            # Aguardar 30 segundos
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 IA³ Calculator stopping...")
            break
        except Exception as e:
            print(f"❌ Error in cycle {cycle}: {e}")
            time.sleep(60)