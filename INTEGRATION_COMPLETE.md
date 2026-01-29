# 🚀 MASTER INTEGRATION GUIDE - Complete Kedro ML Engine

## 📋 Executive Summary

Your Kedro ML Engine is **already fully generic and production-ready** for:
- ✅ Any CSV/Excel dataset
- ✅ Any structured tabular data
- ✅ Classification & Regression problems
- ✅ Any target variable
- ✅ 100% configuration-driven (no code changes needed)

This guide explains the complete integrated system and how to use it with your own datasets.

---

## 📊 Complete Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR DATASET (CSV, Excel, etc)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA LOADING & VALIDATION                              │
│ ├─ load_raw_data() → Loads from data_path                       │
│ ├─ separate_target() → Separates target_column                  │
│ ├─ validate_data() → Checks for issues                          │
│ ├─ clean_data() → Handles missing values                        │
│ └─ split_train_test() → Stratified split with random_state     │
│                                                                  │
│ OUTPUT: X_train_raw, X_test_raw, y_train, y_test               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE ENGINEERING & SELECTION                        │
│ ├─ engineer_features() → Drop IDs, scale, encode               │
│ │  ├─ Detect & drop ID columns (smart cardinality check)       │
│ │  ├─ Scale numeric features                                    │
│ │  ├─ Encode categorical features (one-hot/label)              │
│ │  ├─ Remove low-variance features                              │
│ │  └─ Validate feature count (no explosion)                     │
│ │                                                                │
│ ├─ feature_selection() → SelectKBest or importance-based        │
│ └─ handle_class_imbalance() → SMOTE if needed                   │
│                                                                  │
│ OUTPUT: X_train_selected, X_test_selected                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3: MODEL TRAINING & TUNING                                │
│ ├─ scale_features() → Additional scaling for models             │
│ ├─ train_baseline() → Baseline model                            │
│ ├─ train_best_model() → Hyperparameter tuning                   │
│ ├─ evaluate_model() → Comprehensive metrics                     │
│ └─ cross_validation() → K-fold evaluation                       │
│                                                                  │
│ OUTPUT: baseline_model, best_model, model_evaluation            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: ALGORITHM COMPARISON & ENSEMBLE                        │
│ ├─ Compare: LogisticRegression, RandomForest, GradientBoosting │
│ ├─ Visualize: ROC curves, Confusion matrices, Learning curves   │
│ ├─ Analyze: Statistical tests, SHAP values                      │
│ └─ Ensemble: Voting, Stacking, Blending                         │
│                                                                  │
│ OUTPUT: Algorithm comparison, Ensemble models                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 5: ADVANCED EVALUATION & REPORTING (Python Classes)       │
│ ├─ ComprehensiveMetricsCalculator → 40+ metrics                │
│ ├─ VisualizationManager → 10+ plot types                        │
│ ├─ ComprehensiveReportManager → HTML/JSON/PDF/Model Cards       │
│ └─ CrossValidationStrategies → 6 CV approaches                 │
│                                                                  │
│ USAGE: Python code after Phase 1-4                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 6: ADVANCED ENSEMBLE METHODS                              │
│ ├─ Stacking ensemble (multiple levels)                          │
│ ├─ Blending approach                                            │
│ ├─ Weighted voting                                              │
│ └─ Meta-learner optimization                                    │
│                                                                  │
│ OUTPUT: Final ensemble model with best performance              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                    📊 FINAL RESULTS 📊
```

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Prepare Your Data

```bash
# Copy your CSV file to data/01_raw/
cp /path/to/your/data.csv kedro-ml-engine-integrated/data/01_raw/

# Or use a file that's already there:
# - WA_Fn-UseC_-Telco-Customer-Churn.csv (classification)
# - adult.csv (classification)
```

### Step 2: Update Configuration

Edit `conf/base/parameters.yml`:

```yaml
# ========== DATA PATHS ==========
data_path: "data/01_raw/your_data.csv"     # YOUR FILE
target_column: "your_target_column"         # YOUR TARGET

# ========== DATA PROCESSING ==========
data_processing:
  handle_missing: "mean"                    # mean, median, forward_fill, drop
  test_size: 0.2                            # 80% train, 20% test
  random_state: 42                          # For reproducibility
  stratify: null                            # Set to target_column for classification

# ========== FEATURE ENGINEERING ==========
feature_engineering:
  drop_id_columns: true                     # Auto-detect & drop IDs
  polynomial_features: false                # Disable to prevent explosion
  variance_threshold: 0.01                  # Drop low-variance features
  max_features_allowed: 500                 # Safety limit

categorical:
  encoding_method: "smart"
  max_categories_to_onehot: 10              # Categories > this → drop
  drop_first_category: true

scaling:
  method: "standard"                        # standard, minmax, robust

feature_selection:
  method: "importance"                      # importance, correlation, recursive
  n_features: 20                            # Number of features to select
```

### Step 3: Run Pipeline

```bash
cd kedro-ml-engine-integrated

# Run all phases (1-6)
kedro run

# Or run specific phases
kedro run --pipeline phase1                 # Data loading & validation
kedro run --pipeline phase2                 # Feature engineering & selection
kedro run --pipeline phase3                 # Model training
kedro run --pipeline phase4                 # Algorithm comparison
kedro run --pipeline complete_1_6           # All phases 1-6
```

### Step 4: View Results

```bash
# Check outputs in data/08_reporting/
ls -la data/08_reporting/

# Or view in Kedro UI
kedro viz
```

---

## 📁 Project Structure

```
kedro-ml-engine-integrated/
│
├── conf/                                    # Configuration
│   └── base/
│       ├── parameters.yml                  # All settings here!
│       └── catalog.yml                     # Data definitions
│
├── data/                                    # Data directory
│   ├── 01_raw/                             # Your input CSVs
│   ├── 02_intermediate/                    # Validation & cleaning output
│   ├── 03_primary/                         # Feature engineering output
│   ├── 05_model_input/                     # Model ready data
│   └── 08_reporting/                       # Final reports & visualizations
│
├── src/ml_engine/
│   ├── pipelines/                          # All phase code
│   │   ├── data_loading.py                 # Phase 1a - Load & split
│   │   ├── data_validation.py              # Phase 1b - Validate
│   │   ├── data_cleaning.py                # Phase 1c - Clean
│   │   ├── feature_engineering.py          # Phase 2a - Engineer features
│   │   ├── feature_selection.py            # Phase 2b - Select features
│   │   ├── model_training.py               # Phase 3 - Train models
│   │   ├── phase4_algorithms.py            # Phase 4 - Compare algorithms
│   │   ├── phase6_ensemble_pipeline.py    # Phase 6 - Ensemble methods
│   │   └── [Phase 5 modules]               # Analysis & reporting
│   │
│   └── pipeline_registry.py                # Pipeline orchestration
│
└── notebooks/                              # Example notebooks
    ├── exploratory/                        # Data exploration
    └── reports/                            # Results & analysis
```

---

## 🔄 Complete Data Flow

### Phase 1: Data Loading

**What it does:**
- Loads CSV/Excel from `data_path`
- Separates target column
- Validates data quality
- Handles missing values
- Splits into train/test (stratified if classification)

**Configuration (in parameters.yml):**
```yaml
data_path: "data/01_raw/your_file.csv"
target_column: "your_target"
data_processing:
  test_size: 0.2
  random_state: 42
  stratify: null  # Set to target_column for classification
```

**Outputs:**
- X_train_raw, X_test_raw
- y_train, y_test
- split_summary (metadata)

---

### Phase 2: Feature Engineering

**What it does:**
- Detects & drops ID columns automatically
- Scales numeric features (StandardScaler)
- Encodes categorical features (one-hot or label)
- Removes low-variance features
- Selects best features (SelectKBest or importance-based)
- Handles class imbalance (SMOTE if needed)

**Configuration:**
```yaml
feature_engineering:
  drop_id_columns: true
  polynomial_features: false  # Disable to avoid explosion
  variance_threshold: 0.01    # Drop if variance < this
  max_features_allowed: 500   # Safety limit

categorical:
  max_categories_to_onehot: 10  # > this → drop or label encode
  encoding_method: "smart"

feature_selection:
  method: "importance"        # importance, correlation
  n_features: 20
```

**Outputs:**
- X_train_selected, X_test_selected
- encoder & scaler objects

---

### Phase 3: Model Training

**What it does:**
- Trains baseline model (Logistic Regression or Linear Regression)
- Tunes hyperparameters (RandomizedSearchCV)
- Evaluates with multiple metrics
- Performs cross-validation
- Predicts on test set

**Configuration:**
```yaml
problem_type: "classification"  # or "regression"
feature_selection:
  method: "importance"
  n_features: 20
```

**Outputs:**
- baseline_model, best_model
- model_evaluation (metrics)
- cross_validation_results
- phase3_predictions

---

### Phase 4: Algorithm Comparison & Ensemble

**What it does:**
- Trains multiple algorithms (LogReg, RF, GB, XGB, etc.)
- Compares performance
- Creates ROC curves, confusion matrices
- Performs statistical tests
- Creates ensemble (voting, stacking)

**Outputs:**
- Algorithm comparison report
- Ensemble model
- Visualizations

---

### Phase 5: Advanced Analysis (Python Classes)

**Available modules** (use in Python after pipeline runs):

```python
# 1. Comprehensive Metrics (40+ automatic metrics)
from ml_engine.pipelines.evaluation_metrics import ComprehensiveMetricsCalculator
calc = ComprehensiveMetricsCalculator()
metrics = calc.evaluate_classification(y_test, y_pred, y_proba)

# 2. Visualizations (10+ plot types)
from ml_engine.pipelines.visualization_manager import VisualizationManager
viz = VisualizationManager()
viz.plot_confusion_matrix(y_test, y_pred, 'output.png')
viz.plot_roc_curve(y_test, y_proba)

# 3. Reports (HTML, JSON, PDF, Model Cards)
from ml_engine.pipelines.report_generator import ComprehensiveReportManager
report = ComprehensiveReportManager('MyModel')
reports = report.generate_all_reports('./data/08_reporting')

# 4. Cross-validation strategies
from ml_engine.pipelines.cross_validation_strategies import CVSelector
cv = CVSelector()
results = cv.stratified_kfold(model, X_train, y_train, n_splits=5)
```

---

### Phase 6: Ensemble Methods

**What it does:**
- Stacking ensemble (multiple levels)
- Blending approach
- Weighted voting optimization
- Meta-learner training

**Outputs:**
- Final ensemble model with best performance

---

## 📊 Example: Using with Different Datasets

### Example 1: Telco Customer Churn

```yaml
# conf/base/parameters.yml
data_path: "WA_Fn-UseC_-Telco-Customer-Churn.csv"
target_column: "Churn"

data_processing:
  handle_missing: "mean"
  test_size: 0.2
  random_state: 42
  stratify: "Churn"  # Classification → stratify

feature_engineering:
  drop_id_columns: true
  polynomial_features: false
  max_features_allowed: 200

categorical:
  max_categories_to_onehot: 10
  encoding_method: "smart"

problem_type: "classification"  # Predicting Churn (yes/no)
```

### Example 2: Adult Income (>50K or <=50K)

```yaml
data_path: "adult.csv"
target_column: "income"

data_processing:
  handle_missing: "median"
  test_size: 0.2
  stratify: "income"

feature_engineering:
  drop_id_columns: false  # Adult has no clear IDs
  polynomial_features: false
  max_features_allowed: 150

feature_selection:
  n_features: 15

problem_type: "classification"
```

### Example 3: Home Credit (Custom Dataset)

```yaml
data_path: "data/01_raw/application_train.csv"
target_column: "TARGET"

data_processing:
  handle_missing: "mean"
  test_size: 0.3  # 70/30 split
  stratify: "TARGET"

feature_engineering:
  drop_id_columns: true
  id_keywords: ["id", "sk_id", "account"]
  polynomial_features: false
  max_features_allowed: 300

categorical:
  max_categories_to_onehot: 15
  
feature_selection:
  method: "importance"
  n_features: 25

problem_type: "classification"
```

---

## 🔧 Configuration Options Explained

### data_path
```yaml
data_path: "data/01_raw/your_file.csv"
# Supports:
# - CSV files: "path/to/file.csv"
# - Excel files: "path/to/file.xlsx"
# - Relative paths (from project root)
```

### target_column
```yaml
target_column: "your_target_column"
# Must be a column in your dataset
# This is what you're trying to predict
```

### handle_missing
```yaml
handle_missing: "mean"    # Numeric features: fill with mean
handle_missing: "median"  # Numeric features: fill with median
handle_missing: "forward_fill"  # Time series: forward fill
handle_missing: "drop"    # Drop rows with missing values
```

### stratify
```yaml
stratify: null                  # Random split (regression or imbalanced)
stratify: "target_column"       # Stratified split (classification)
# Stratified = preserves class distribution in train/test
```

### polynomial_features
```yaml
polynomial_features: false  # Don't create poly features (recommended)
polynomial_features: true   # Create X^2, X^3, etc. (use with caution!)
# Warning: Can cause feature explosion!
```

### encoding_method
```yaml
encoding_method: "smart"    # Auto-decide (one-hot if < max_categories_to_onehot, else label)
encoding_method: "one_hot"  # Always one-hot encode
encoding_method: "label"    # Always label encode
```

### feature_selection
```yaml
feature_selection:
  method: "importance"      # Tree-based feature importance
  method: "correlation"     # Correlation with target
  method: "recursive"       # Recursive feature elimination
  n_features: 20            # Select top 20 features
```

### problem_type
```yaml
problem_type: "classification"  # Binary/multiclass classification
problem_type: "regression"      # Continuous target
# Determines which models to train and which metrics to report
```

---

## ✅ Checklist: Using with Your Own Dataset

- [ ] Copy your CSV/Excel file to `data/01_raw/`
- [ ] Update `data_path` in `conf/base/parameters.yml`
- [ ] Update `target_column` to your target variable
- [ ] Review other parameters (adjust if needed)
- [ ] Check if `stratify` should be set (classification)
- [ ] Review `max_categories_to_onehot` (adjust for your data)
- [ ] Run `kedro run --pipeline phase1` to test data loading
- [ ] Run `kedro run` to run full pipeline
- [ ] Check `data/08_reporting/` for results

---

## 📊 Running Phases Individually

```bash
# Phase 1: Data loading & validation
kedro run --pipeline phase1
# Outputs: X_train_raw, X_test_raw, y_train, y_test

# Phase 2: Feature engineering & selection
kedro run --pipeline phase2
# Outputs: X_train_selected, X_test_selected

# Phase 3: Model training
kedro run --pipeline phase3
# Outputs: baseline_model, best_model, evaluation

# Phase 4: Algorithm comparison & ensemble
kedro run --pipeline phase4
# Outputs: Algorithm comparison, ensemble model

# Complete pipeline (Phase 1-4)
kedro run --pipeline complete

# With ensemble (Phase 1-6)
kedro run --pipeline complete_1_6
```

---

## 🎨 Visualizations & Reports

After running the pipeline, use Phase 5 modules:

```python
import pandas as pd
import pickle
from ml_engine.pipelines.visualization_manager import VisualizationManager
from ml_engine.pipelines.report_generator import ComprehensiveReportManager

# Load results from pipeline
with open('data/08_reporting/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_test = pd.read_csv('data/03_primary/y_test.csv').iloc[:, 0]
y_pred = model.predict(X_test)

# Create visualizations
viz = VisualizationManager()
viz.plot_confusion_matrix(y_test, y_pred)
viz.plot_roc_curve(y_test, y_proba)
viz.plot_feature_importance(model)

# Generate comprehensive report
report = ComprehensiveReportManager('MyModel')
reports = report.generate_all_reports('./data/08_reporting')
```

---

## 🔍 Troubleshooting

### Issue: "Target column not found"
**Solution:** Check spelling in `parameters.yml`. Column names are case-sensitive!

### Issue: "No data loaded"
**Solution:** Verify `data_path` exists and is readable. Check file format (CSV/Excel).

### Issue: "Too many features created"
**Solution:** In `parameters.yml`, set:
```yaml
polynomial_features: false  # Disable polynomial features
max_categories_to_onehot: 5  # Lower threshold for one-hot
max_features_allowed: 200    # Lower limit
```

### Issue: "Models perform poorly"
**Solution:** 
- Check if data is scaled correctly: `scaling.method: "standard"`
- Check if features are relevant: Inspect feature_selection
- Try different `problem_type`: "classification" or "regression"

---

## 📚 Complete File Structure for Reference

```
conf/base/
├── parameters.yml           ← Edit this for your data!
├── catalog.yml
└── settings.yml

data/01_raw/
└── your_dataset.csv         ← Put your data here

data/03_primary/
├── X_train_raw.csv          ← Phase 1 output
├── X_test_raw.csv
├── y_train.csv
└── y_test.csv

data/08_reporting/
├── model_evaluation.pkl     ← Phase 3 results
├── algorithm_comparison.pkl ← Phase 4 results
└── reports/                 ← Final reports
```

---

## 🚀 Next Steps

1. **Prepare your data** → Copy CSV to `data/01_raw/`
2. **Update parameters** → Edit `conf/base/parameters.yml`
3. **Run pipeline** → `kedro run`
4. **Review results** → Check `data/08_reporting/`
5. **Generate reports** → Use Phase 5 Python classes
6. **Deploy model** → Use best_model from Phase 4 output

---

## 💡 Key Principles

✅ **Configuration-Driven:** No code changes needed for different datasets  
✅ **Production-Ready:** Error handling, logging, validation at every step  
✅ **Generic:** Works with any CSV/Excel tabular data  
✅ **Reproducible:** Fixed random_state ensures same results every run  
✅ **Safe:** Prevents feature explosion with automatic limits  
✅ **Comprehensive:** 6 phases of analysis & reporting  
✅ **Modular:** Run phases individually or together  

---

**Your ML pipeline is ready to use with ANY dataset!** 🎉

Just update `parameters.yml` and run `kedro run`. That's it!
