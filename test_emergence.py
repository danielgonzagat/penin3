#!/usr/bin/env python3
import requests
import time
import json

def test_emergence():
    print("🧪 Testando emergência...")
    try:
        response = requests.get('http://127.0.0.1:9100/metrics', timeout=10)
        if response.status_code == 200:
            metrics = response.text
            if 'emergence_probability' in metrics:
                prob = float(metrics.split('emergence_probability ')[1].split('\n')[0])
                print(f"✅ Emergência detectada: prob={prob:.3f}")
                return prob > 0
            else:
                print("❌ Métrica emergence_probability ausente")
                return False
        else:
            print(f"❌ Falha no endpoint: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    if test_emergence():
        print("🎉 Inteligência emergente ativa!")
    else:
        print("😔 Sem emergência detectada.")
