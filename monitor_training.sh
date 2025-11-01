#!/bin/bash
# Monitor training progress on remote server

# Configuration - EDIT THESE FOR YOUR SETUP
REMOTE_HOST="${REMOTE_HOST:-user@remote-server.com}"
REMOTE_DIR="${REMOTE_DIR:-~/text-diffusion}"

echo "================================================================================
TEXT DIFFUSION TRAINING MONITOR
================================================================================
"

# Check if training is running
if ssh $REMOTE_HOST "screen -ls" | grep -q "text-diffusion-training"; then
    echo "✅ Training session is ACTIVE"
else
    echo "❌ Training session NOT FOUND"
    exit 1
fi

echo ""
echo "Training Progress:"
echo "--------------------------------------------------------------------------------"

# Get latest losses
echo ""
echo "Recent Loss Values:"
ssh $REMOTE_HOST "cd $REMOTE_DIR && grep 'loss' training.log | tail -10 | sed 's/^/  /'"

echo ""
echo "--------------------------------------------------------------------------------"

# Count total steps
TOTAL_STEPS=4458
CURRENT_STEP=$(ssh $REMOTE_HOST "cd $REMOTE_DIR && grep -oP 'epoch\': \K[0-9.]+' training.log | tail -1" 2>/dev/null || echo "0")

if [ -n "$CURRENT_STEP" ] && [ "$CURRENT_STEP" != "0" ]; then
    CURRENT_STEP_NUM=$(echo "$CURRENT_STEP * 1486" | bc)
    PERCENT=$(echo "scale=1; $CURRENT_STEP * 100 / 3" | bc)
    echo "Progress: ${PERCENT}% of total training"
    echo "Epoch: ${CURRENT_STEP} / 3.0"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "Latest Output:"
ssh $REMOTE_HOST "cd $REMOTE_DIR && tail -20 training.log | grep -E 'it/s|loss' | tail -5 | sed 's/^/  /'"

echo ""
echo "================================================================================
Commands:
  Watch live:   ssh $REMOTE_HOST 'tail -f $REMOTE_DIR/training.log'
  Attach screen: ssh $REMOTE_HOST 'screen -r text-diffusion-training'
  Check checkpoints: ssh $REMOTE_HOST 'ls -lh $REMOTE_DIR/results-full/'
================================================================================
"
