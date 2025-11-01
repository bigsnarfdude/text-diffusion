#!/bin/bash
# Monitor training progress on nigel.birs.ca

echo "================================================================================
TEXT DIFFUSION TRAINING MONITOR
================================================================================
"

# Check if training is running
if ssh vincent@nigel.birs.ca "screen -ls" | grep -q "text-diffusion-training"; then
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
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && grep 'loss' training.log | tail -10 | sed 's/^/  /'"

echo ""
echo "--------------------------------------------------------------------------------"

# Count total steps
TOTAL_STEPS=4458
CURRENT_STEP=$(ssh vincent@nigel.birs.ca "cd ~/text-diffusion && grep -oP 'epoch\': \K[0-9.]+' training.log | tail -1" 2>/dev/null || echo "0")

if [ -n "$CURRENT_STEP" ] && [ "$CURRENT_STEP" != "0" ]; then
    CURRENT_STEP_NUM=$(echo "$CURRENT_STEP * 1486" | bc)
    PERCENT=$(echo "scale=1; $CURRENT_STEP * 100 / 3" | bc)
    echo "Progress: ${PERCENT}% of total training"
    echo "Epoch: ${CURRENT_STEP} / 3.0"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "Latest Output:"
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && tail -20 training.log | grep -E 'it/s|loss' | tail -5 | sed 's/^/  /'"

echo ""
echo "================================================================================
Commands:
  Watch live:   ssh vincent@nigel.birs.ca 'tail -f ~/text-diffusion/training.log'
  Attach screen: ssh vincent@nigel.birs.ca 'screen -r text-diffusion-training'
  Check checkpoints: ssh vincent@nigel.birs.ca 'ls -lh ~/text-diffusion/results-full/'
================================================================================
"
