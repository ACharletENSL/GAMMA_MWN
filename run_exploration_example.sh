#!/bin/bash
# Example script to run parameter space exploration

echo "=================================================="
echo "Parameter Space Exploration - Example Run"
echo "=================================================="
echo ""

# Check prerequisites
echo "1. Checking prerequisites..."
if [ ! -f "bin/GAMMA" ]; then
    echo "   ✗ GAMMA executable not found. Please compile with 'make -B'"
    exit 1
fi
echo "   ✓ GAMMA executable found"

if [ ! -d "results/Last" ]; then
    echo "   Creating results/Last directory..."
    mkdir -p results/Last
fi
echo "   ✓ results/Last directory exists"

# Run quick test first
echo ""
echo "2. Running quick test (4 simulations)..."
echo "   This will take approximately 30-60 minutes"
python run_quick_exploration.py

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "   ✓ Quick test completed"
    echo ""
    echo "3. Test results saved to: quick_test_results.csv"
    
    # Ask user if they want to continue with full exploration
    echo ""
    echo "4. Do you want to run the full parameter space exploration?"
    echo "   (This may take several hours depending on your parameter ranges)"
    read -p "   Continue? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "5. Running full parameter space exploration..."
        python parameter_space_exploration.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "=================================================="
            echo "Exploration completed successfully!"
            echo "=================================================="
            echo ""
            echo "Results saved to: parameter_space_results.csv"
            echo ""
            echo "Next steps:"
            echo "  1. View results: open parameter_space_results.csv"
            echo "  2. Generate plots: python visualize_parameter_space.py"
            echo "  3. Read documentation: parameter_exploration_README.md"
            echo ""
        else
            echo ""
            echo "✗ Full exploration failed. Check error messages above."
            exit 1
        fi
    else
        echo ""
        echo "Full exploration cancelled by user."
        echo "You can run it later with: python parameter_space_exploration.py"
        exit 0
    fi
else
    echo ""
    echo "✗ Quick test failed. Please check error messages above."
    echo ""
    echo "Common issues:"
    echo "  - GAMMA not compiled: run 'make -B'"
    echo "  - Missing Python packages: pip install numpy pandas scipy"
    echo "  - MPI not available: check 'which mpirun'"
    exit 1
fi

echo "=================================================="
