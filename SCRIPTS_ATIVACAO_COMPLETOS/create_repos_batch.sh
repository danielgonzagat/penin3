#!/bin/bash
TOKEN="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"

# Repo 4: Agent Behavior Learner IA³
curl -H "Authorization: token $TOKEN" -d '{"name":"agent-behavior-learner-ia3","description":"IA³ Agent Behavior Learner - Pattern discovery without hardcoded learning, clustering with DBSCAN/KMeans","private":false}' https://api.github.com/user/repos &

# Repo 5: AGI Singularity Emergent Real
curl -H "Authorization: token $TOKEN" -d '{"name":"agi-singularity-emergent-real","description":"AGI with Real Self-Modification - AI that programs AI using real tools","private":false}' https://api.github.com/user/repos &

# Repo 6: Intelligence Emergence Maestro
curl -H "Authorization: token $TOKEN" -d '{"name":"intelligence-emergence-maestro","description":"Top 10 Systems Coordinator - Real anomaly detection with DBSCAN, orchestrates emergence","private":false}' https://api.github.com/user/repos &

# Repo 7: THE NEEDLE Evolved Meta
curl -H "Authorization: token $TOKEN" -d '{"name":"needle-evolved-meta","description":"Meta-Learning System - Learns to learn, curriculum: MNIST→CIFAR→RL","private":false}' https://api.github.com/user/repos &

wait
echo "All 4 repos created!"
