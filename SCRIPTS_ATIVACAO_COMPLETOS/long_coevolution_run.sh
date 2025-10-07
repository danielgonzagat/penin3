#!/bin/bash

# Long controlled TEIS↔PUES coevolution run with periodic snapshots
# Will run 30 loops with snapshots every 5 loops

echo "======================================"
echo "LONG COEVOLUTION AUDIT RUN"
echo "Starting at: $(date)"
echo "======================================"

# Create snapshot directory
mkdir -p coevolution_snapshots

# Initialize loop counter
LOOP=0
MAX_LOOPS=30
SNAPSHOT_EVERY=5

# Main evolution loop
while [ $LOOP -lt $MAX_LOOPS ]; do
    LOOP=$((LOOP + 1))
    
    echo ""
    echo "========== LOOP $LOOP/$MAX_LOOPS =========="
    echo "Time: $(date +%H:%M:%S)"
    
    # Run one evolution cycle (60 seconds TEIS, 3 PUES generations)
    python3 perpetual_evolution_manager.py \
        --loop-seconds 60 \
        --sleep-seconds 5 \
        --pues-gens 3 \
        --optimize-every 1 \
        --max-loops 1 \
        2>&1 | tee -a coevolution_audit.log
    
    # Take snapshot every N loops
    if [ $((LOOP % SNAPSHOT_EVERY)) -eq 0 ]; then
        echo "📸 Taking snapshot at loop $LOOP..."
        
        # Create snapshot directory for this loop
        SNAPSHOT_DIR="coevolution_snapshots/loop_${LOOP}"
        mkdir -p "$SNAPSHOT_DIR"
        
        # Copy key metrics files
        cp emergent_metrics.json "$SNAPSHOT_DIR/" 2>/dev/null || echo "{}" > "$SNAPSHOT_DIR/emergent_metrics.json"
        cp emergent_insights.json "$SNAPSHOT_DIR/" 2>/dev/null || echo "{}" > "$SNAPSHOT_DIR/emergent_insights.json"
        cp teis_emergent_fitness.json "$SNAPSHOT_DIR/" 2>/dev/null || echo "{}" > "$SNAPSHOT_DIR/teis_emergent_fitness.json"
        cp perpetual_status.json "$SNAPSHOT_DIR/" 2>/dev/null || echo "{}" > "$SNAPSHOT_DIR/perpetual_status.json"
        
        # Count emergent behaviors
        BEHAVIOR_COUNT=$(wc -l emergent_behaviors_log.jsonl 2>/dev/null | awk '{print $1}' || echo "0")
        echo "$BEHAVIOR_COUNT" > "$SNAPSHOT_DIR/behavior_count.txt"
        
        # Log snapshot
        echo "{\"loop\": $LOOP, \"time\": \"$(date -Iseconds)\", \"behaviors\": $BEHAVIOR_COUNT}" >> coevolution_snapshots/timeline.jsonl
    fi
    
    # Brief pause between loops
    sleep 2
done

echo ""
echo "======================================"
echo "LONG COEVOLUTION AUDIT COMPLETE"
echo "Finished at: $(date)"
echo "Total loops: $MAX_LOOPS"
echo "======================================"

# Final snapshot
echo "📸 Taking final snapshot..."
cp emergent_metrics.json coevolution_snapshots/final_emergent_metrics.json 2>/dev/null
cp emergent_insights.json coevolution_snapshots/final_emergent_insights.json 2>/dev/null
cp teis_emergent_fitness.json coevolution_snapshots/final_teis_fitness.json 2>/dev/null
cp perpetual_status.json coevolution_snapshots/final_perpetual_status.json 2>/dev/null

# Final behavior count
FINAL_BEHAVIORS=$(wc -l emergent_behaviors_log.jsonl 2>/dev/null | awk '{print $1}' || echo "0")
echo "Final emergent behaviors: $FINAL_BEHAVIORS"
echo "$FINAL_BEHAVIORS" > coevolution_snapshots/final_behavior_count.txt