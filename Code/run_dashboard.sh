#!/usr/bin/env bash
# ==============================================================================
# Script: Code/run_dashboard.sh
# Purpose: Launch the Streamlit DDOG Nowcasting Dashboard
# ==============================================================================

# Ensure execution starts in the Code directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========================================================"
echo " Starting Datadog (DDOG) Revenue Nowcasting Dashboard..."
echo "========================================================"

# Check if Python is available
if command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Error: Python not found in PATH."
    exit 1
fi

# Run Streamlit dashboard
$PYTHON_CMD -m streamlit run dashboard.py
