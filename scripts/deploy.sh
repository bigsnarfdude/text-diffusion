#!/bin/bash
# Deploy text-diffusion project to remote server

set -e

# Configuration - EDIT THESE FOR YOUR SETUP
REMOTE_HOST="${REMOTE_HOST:-user@remote-server.com}"
REMOTE_DIR="${REMOTE_DIR:-~/text-diffusion}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================="
echo "Deploying to $REMOTE_HOST"
echo "=================================="

# Check if remote host is configured
if [[ "$REMOTE_HOST" == "user@remote-server.com" ]]; then
    echo "⚠️  Error: Please configure REMOTE_HOST"
    echo "Edit this script or set environment variable:"
    echo "  export REMOTE_HOST=your-user@your-server.com"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Local: $LOCAL_DIR"
echo "  Remote: $REMOTE_HOST:$REMOTE_DIR"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 $REMOTE_HOST "echo 'Connection successful'"; then
    echo "❌ Cannot connect to $REMOTE_HOST"
    echo "Check SSH keys and network connection"
    exit 1
fi
echo "✅ SSH connection verified"

# Create remote directory if needed
echo ""
echo "Creating remote directory..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Copy files
echo ""
echo "Copying files..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results/' \
    --exclude 'experiments/*.png' \
    $LOCAL_DIR/ $REMOTE_HOST:$REMOTE_DIR/

echo ""
echo "✅ Files copied successfully"

# Install dependencies
echo ""
echo "Installing dependencies on nigel..."
ssh $REMOTE_HOST "cd $REMOTE_DIR && python3 -m pip install --user -q torch transformers datasets accelerate matplotlib 2>&1 | grep -i 'success\|error\|warning' || true"

# Test installation
echo ""
echo "Testing installation..."
ssh $REMOTE_HOST "cd $REMOTE_DIR && python3 -c 'import torch, transformers, datasets; print(\"✅ All packages imported successfully\")'"

# Show next steps
echo ""
echo "=================================="
echo "Deployment Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. SSH to nigel:"
echo "   ssh $REMOTE_HOST"
echo ""
echo "2. Navigate to project:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Run visualization experiments:"
echo "   python3 experiments/masking_viz.py"
echo ""
echo "4. Quick test training:"
echo "   python3 train.py --quick-test"
echo ""
echo "5. Generate text:"
echo "   python3 generate.py --checkpoint results/final-model --prefix \"Machine learning is\""
echo ""
echo "See QUICKSTART.md for detailed guide"
echo ""
