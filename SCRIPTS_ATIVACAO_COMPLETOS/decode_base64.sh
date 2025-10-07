#!/bin/bash
echo "📝 Cole o conteúdo base64 do arquivo e pressione Ctrl+D:"
cat > /tmp/github_backup_b64.txt

echo "🔄 Decodificando arquivo..."
base64 -d /tmp/github_backup_b64.txt > /root/github_backup.zip

if [ -f "/root/github_backup.zip" ]; then
    echo "✅ Arquivo decodificado com sucesso!"
    ls -la /root/github_backup.zip
else
    echo "❌ Erro na decodificação"
fi
