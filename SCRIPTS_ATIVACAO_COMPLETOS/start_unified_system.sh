#!/bin/bash
# start_unified_system.sh
# Orchestrator script to launch all IA³ subsystems as independent background processes.

echo "======================================================"
echo " IA³ UNIFIED INTELLIGENCE SYSTEM - LAUNCHING..."
echo "======================================================"

# Create log directory
LOG_DIR="/root/cns_logs"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

# Clean up previous trace bus
rm -f /root/cns_data/trace_bus.jsonl
echo "Cleared previous trace bus."

# List of python runners to launch
runners=(
    "run_perception.py"
    "run_environment.py"
    "run_consciousness.py"
    "run_arch_opt.py"
    "run_semantic.py"
    "run_teis.py"
)

# Launch each runner as a background process
for runner in "${runners[@]}"; do
    log_file="$LOG_DIR/$(basename "$runner" .py).log"
    echo "Launching $runner... Log file: $log_file"
    # Use nohup to ensure processes keep running even if the shell is closed
    # Redirect both stdout and stderr to the log file
    nohup python3 "/root/$runner" > "$log_file" 2>&1 &
    # Small delay to prevent race conditions during initialization
    sleep 2
done

echo "------------------------------------------------------"
echo "All subsystems have been launched in the background."
echo "Use 'ps aux | grep python3' to see the running processes."
echo "Use 'tail -f $LOG_DIR/<module_name>.log' to monitor a specific module."
echo "Use 'tail -f /root/cns_data/trace_bus.jsonl' to monitor the CNS."
echo "SYSTEM IS NOW LIVE."
echo "======================================================"
