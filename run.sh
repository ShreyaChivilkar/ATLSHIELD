#!/bin/bash

# Create virtual environment
if [ ! -d "venv" ]; then
  echo "🔧 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run the app with nohup
echo "🚀 Starting atlShield.py using nohup..."
nohup python3 atlShield.py > atlShield.log 2>&1 &

echo "atlShield is now running in the background. Logs: atlShield.log"
