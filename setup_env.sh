#!/bin/bash

# Exit immediately on error
set -e

# Set your virtual environment directory
VENV_DIR=".venv"

# Create virtual environment
echo "Creating virtual environment with Python 3.12.2..."
python -m venv $VENV_DIR

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "✅ Environment setup complete."
echo "To activate the virtual environment later, run:"
echo "source $VENV_DIR/bin/activate"
