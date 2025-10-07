#!/usr/bin/env python3
import requests, time, json

def check_emergence():
    try:
        r = requests.get('http://127.0.0.1:9100/metrics', timeout=5)
        if r.status_code == 200:
            metrics = r.text
            if 'emergence_probability' in metrics:
                prob = float(metrics.split('emergence_probability ')[1].split('\n')[0])
                if prob > 0:
                    print(f"🎉 Emergência detectada! Probabilidade: {prob:.3f}")
                    return True
    except Exception:
        pass
    return False

while True:
    if check_emergence():
        break
    time.sleep(60)
