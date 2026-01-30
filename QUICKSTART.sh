#!/bin/bash
# QUICKSTART GUIDE
# Run this to set up and test the complete autoscaling pipeline

echo "=================================="
echo "DataFlow 2026 - Autoscaling Pipeline"
echo "Quick Start Setup"
echo "=================================="
echo

# Step 1: Check Python
echo "✓ Checking Python environment..."
python --version

# Step 2: Install dependencies
echo
echo "✓ Installing dependencies..."
pip install pandas numpy scikit-learn statsmodels plotly streamlit

# Step 3: Run simulation
echo
echo "✓ Running full autoscaling simulation..."
echo "  (This will test 5 scenarios × 4 strategies = 20 experiments)"
python simulate.py

# Step 4: Check results
echo
echo "✓ Results generated in ./results/"
ls -lh results/

echo
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo
echo "Next steps:"
echo
echo "1. View detailed results:"
echo "   $ head -5 results/simulation_results.csv"
echo
echo "2. See metrics summary:"
echo "   $ cat results/metrics_summary.json | head -50"
echo
echo "3. Launch interactive dashboard:"
echo "   $ streamlit run dashboard/app.py"
echo "   (Opens at http://localhost:8501)"
echo
echo "4. Read full documentation:"
echo "   $ cat README.md"
echo "   $ cat AUDIT_REPORT.md"
echo
