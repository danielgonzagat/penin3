#!/bin/bash
set -e
TOKEN="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"

# Função para processar um repo
push_repo() {
    local dir=$1
    local name=$2
    local desc=$3
    
    cd "/root/_export_preservation_2025/$dir"
    
    # Git init se necessário
    [ ! -d ".git" ] && git init
    
    # Add e commit
    git add .
    git commit -m "$desc" || echo "Nothing to commit in $name"
    
    # Add remote se necessário
    git remote add origin "https://danielgonzagat:$TOKEN@github.com/danielgonzagat/$name.git" 2>/dev/null || true
    
    # Push
    git branch -M main
    git push -u origin main --force
    
    echo "✅ $name pushed!"
}

# Repo 4
push_repo "agent_behavior_ia3" "agent-behavior-learner-ia3" "Agent Behavior Learner IA³ - Pattern discovery without hardcoded learning" &

# Repo 5  
push_repo "agi_singularity" "agi-singularity-emergent-real" "AGI Singularity - Real self-modification using OS tools" &

# Repo 6
push_repo "emergence_maestro" "intelligence-emergence-maestro" "Intelligence Emergence Maestro - Top 10 coordinator with real anomaly detection" &

# Repo 7
push_repo "needle_meta" "needle-evolved-meta" "THE NEEDLE Evolved Meta - Meta-learning with curriculum MNIST→CIFAR→RL" &

wait
echo "🎉 All 4 repos pushed successfully!"
