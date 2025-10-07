#!/bin/bash

echo "🚀 CONFIGURAÇÃO INTERATIVA DO GITHUB"
echo "===================================="

# Verificar se já existe configuração
echo "📊 Configuração atual do Git:"
git config --global user.name 2>/dev/null || echo "  Nome: Não configurado"
git config --global user.email 2>/dev/null || echo "  Email: Não configurado"

echo ""
echo "🔧 VAMOS CONFIGURAR SUA CONEXÃO COM GITHUB"
echo ""

# Solicitar informações do usuário
read -p "📝 Digite seu nome de usuário do GitHub: " GITHUB_USER
read -p "📧 Digite seu email do GitHub: " GITHUB_EMAIL

# Configurar Git
echo "⚙️ Configurando Git..."
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"

# Gerar chave SSH
echo "🔑 Gerando chave SSH para GitHub..."
ssh-keygen -t ed25519 -C "$GITHUB_EMAIL" -f ~/.ssh/id_ed25519_github -N ""

# Configurar SSH
echo "🔧 Configurando SSH..."
cat > ~/.ssh/config << EOL
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    Port 22
    
Host github-443
    HostName ssh.github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    Port 443
EOL

# Corrigir permissões
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/id_ed25519_github
chmod 644 ~/.ssh/id_ed25519_github.pub

# Mostrar chave pública
echo ""
echo "🔑 SUA CHAVE PÚBLICA SSH (copie e adicione ao GitHub):"
echo "=================================================="
cat ~/.ssh/id_ed25519_github.pub
echo "=================================================="
echo ""

echo "📋 PRÓXIMOS PASSOS:"
echo "1. Copie a chave acima"
echo "2. Acesse: https://github.com/settings/keys"
echo "3. Clique 'New SSH key'"
echo "4. Cole a chave e salve"
echo "5. Execute: ssh -T git@github.com"
echo ""

# Oferecer teste de conexão
read -p "🧪 Quer testar a conexão agora? (y/n): " TEST_NOW

if [[ $TEST_NOW == "y" || $TEST_NOW == "Y" ]]; then
    echo "🧪 Testando conexão..."
    ssh -T git@github.com -o ConnectTimeout=10
fi

echo "✅ Configuração concluída!"
