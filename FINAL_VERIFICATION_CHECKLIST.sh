#!/bin/bash

echo "════════════════════════════════════════════════════════════════════════════"
echo "🔍 FINAL VERIFICATION CHECKLIST"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check 1: Parameters file
echo "✓ CHECK 1: Parameters YAML"
echo "─────────────────────────"
if python -c "import yaml; yaml.safe_load(open('conf/base/parameters.yml'))" 2>/dev/null; then
    echo "✅ parameters.yml is valid YAML"
    echo ""
    echo "   Mode:"
    grep "mode:" conf/base/parameters.yml | head -1
    echo ""
    echo "   Data directory:"
    grep "data_directory:" conf/base/parameters.yml
    echo ""
    echo "   Main table:"
    grep "main_table:" conf/base/parameters.yml
    echo ""
    echo "   Target:"
    grep "target_column:" conf/base/parameters.yml
else
    echo "❌ parameters.yml has YAML errors!"
    exit 1
fi
echo ""

# Check 2: CSV files exist
echo "✓ CHECK 2: CSV Files Location"
echo "─────────────────────────────"
DATA_DIR=$(grep "data_directory:" conf/base/parameters.yml | grep -o '"[^"]*"' | tr -d '"' | head -1)
echo "   Looking in: $DATA_DIR"
echo ""

if [ -d "$DATA_DIR" ]; then
    echo "✅ Directory exists!"
    echo ""
    echo "   Files found:"
    ls -lh "$DATA_DIR" | grep -E "\.csv|\.xlsx" | awk '{print "   • " $9 " (" $5 ")"}'
    echo ""
    
    # Count CSV files
    CSV_COUNT=$(ls "$DATA_DIR"/*.csv 2>/dev/null | wc -l)
    echo "   Total CSV files: $CSV_COUNT"
    
    if [ $CSV_COUNT -lt 2 ]; then
        echo "   ⚠️  WARNING: Expected at least 6 tables for multi-table mode!"
    fi
else
    echo "❌ Directory NOT found: $DATA_DIR"
    exit 1
fi
echo ""

# Check 3: Data loading module
echo "✓ CHECK 3: Data Loading Module"
echo "───────────────────────────────"
if [ -f "src/ml_engine/pipelines/data_loading.py" ]; then
    echo "✅ data_loading.py exists"
    
    if grep -q "def create_pipeline" src/ml_engine/pipelines/data_loading.py; then
        echo "✅ create_pipeline() function exists"
    else
        echo "❌ create_pipeline() NOT found!"
        exit 1
    fi
    
    if grep -q "def load_data_auto" src/ml_engine/pipelines/data_loading.py; then
        echo "✅ load_data_auto() function exists"
    else
        echo "❌ load_data_auto() NOT found!"
        exit 1
    fi
    
    if grep -q "from .data_loading_multitable import" src/ml_engine/pipelines/data_loading/data_loading.py; then
        echo "✅ Imports from data_loading_multitable"
    else
        echo "❌ Does NOT import from data_loading_multitable!"
    fi
else
    echo "❌ data_loading.py NOT found!"
    exit 1
fi
echo ""

# Check 4: Multi-table loader
echo "✓ CHECK 4: Multi-Table Loader"
echo "──────────────────────────────"
if [ -f "src/ml_engine/pipelines/data_loading/data_loading_multitable.py" ]; then
    echo "✅ data_loading_multitable.py exists"
    
    if grep -q "class MultiTableDataLoader" src/ml_engine/pipelines/data_loading/data_loading_multitable.py; then
        echo "✅ MultiTableDataLoader class exists"
    else
        echo "❌ MultiTableDataLoader class NOT found!"
    fi
else
    echo "❌ data_loading_multitable.py NOT found!"
fi
echo ""

# Check 5: Python import test
echo "✓ CHECK 5: Python Import Test"
echo "──────────────────────────────"
python << 'PYTHON'
try:
    from src.ml_engine.pipelines.data_loading import create_pipeline
    p = create_pipeline()
    print(f"✅ Import successful!")
    print(f"✅ Pipeline created with {len(p.nodes)} nodes")
    print(f"✅ Node names: {[n.name for n in p.nodes]}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
PYTHON
echo ""

# Check 6: Data loading configuration
echo "✓ CHECK 6: Configuration Details"
echo "─────────────────────────────────"
python << 'PYTHON'
import yaml

with open('conf/base/parameters.yml') as f:
    params = yaml.safe_load(f)

cfg = params.get('data_loading', {})
mode = cfg.get('mode', 'single')

print(f"✅ Mode: {mode}")

if mode == 'multi':
    print(f"✅ Tables: {len(cfg.get('tables', []))}")
    print(f"✅ Aggregations: {len(cfg.get('aggregations', []))}")
    print(f"✅ Joins: {len(cfg.get('joins', []))}")
    
    print("\n   Tables defined:")
    for t in cfg.get('tables', []):
        print(f"   • {t['name']} ({t['filepath']})")
else:
    print(f"   Filepath: {cfg.get('filepath')}")
PYTHON
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ VERIFICATION COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════"

