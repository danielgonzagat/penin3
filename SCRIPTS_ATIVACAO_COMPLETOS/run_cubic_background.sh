#!/bin/bash
cd /root
exec python3 cubic_farm_24_7.py >> /root/cubic_24_7_logs/background.log 2>&1
