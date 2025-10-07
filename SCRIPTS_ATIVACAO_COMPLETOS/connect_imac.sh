#!/bin/bash
# Script para conectar no iMac do Daniel
echo "Conectando no iMac..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 danielpenin@192.168.1.71
