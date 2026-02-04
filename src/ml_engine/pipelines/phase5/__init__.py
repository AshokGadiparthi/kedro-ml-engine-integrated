"""Phase 5: Advanced Analysis & Reporting Modules

Hybrid module providing both:
1. Kedro pipeline nodes for automated execution
2. Standalone Python classes for manual use
"""

# Main pipeline and utility functions
from .phase5_analysis_pipeline import create_pipeline

# Module 5a: Training Strategies
try:
    from .training_strategies import StratifiedTrainer
except ImportError:
    StratifiedTrainer = None

# Module 5b: Evaluation Metrics
try:
    from .evaluation_metrics import ComprehensiveMetricsCalculator
except ImportError:
    ComprehensiveMetricsCalculator = None

# Module 5c: Cross-Validation Strategies
try:
    from .cross_validation_strategies import (
        CrossValidationComparison,
        StratifiedKFoldCV,
        TimeSeriesCV,
        GroupKFoldCV,
        LeaveOneOutCV,
        ShuffleSplitCV,
        RepeatedStratifiedKFoldCV,
        NestedCV,
    )
except ImportError:
    CrossValidationComparison = None
    StratifiedKFoldCV = None
    TimeSeriesCV = None
    GroupKFoldCV = None
    LeaveOneOutCV = None
    ShuffleSplitCV = None
    RepeatedStratifiedKFoldCV = None
    NestedCV = None

# Module 5d: Model Comparison
try:
    from .model_comparison import ModelComparison
except ImportError:
    ModelComparison = None

# Module 5e: Visualization Manager
try:
    from .visualization_manager import VisualizationManager
except ImportError:
    VisualizationManager = None

# Module 5f: Hyperparameter Analysis
try:
    from .hyperparameter_analysis import (
        ComprehensiveHyperparameterAnalyzer,
        SensitivityAnalysis,
        GeneralizationGapAnalysis,
        ParameterImportance,
        OptimizationSuggestions,
        LearningDynamicsAnalysis,
    )
except ImportError:
    ComprehensiveHyperparameterAnalyzer = None
    SensitivityAnalysis = None
    GeneralizationGapAnalysis = None
    ParameterImportance = None
    OptimizationSuggestions = None
    LearningDynamicsAnalysis = None

# Module 5g: Report Generator
try:
    from .report_generator import ComprehensiveReportManager
except ImportError:
    ComprehensiveReportManager = None

__all__ = [
    # Pipeline
    "create_pipeline",
    # Module 5a
    "StratifiedTrainer",
    # Module 5b
    "ComprehensiveMetricsCalculator",
    # Module 5c
    "CrossValidationComparison",
    "StratifiedKFoldCV",
    "TimeSeriesCV",
    "GroupKFoldCV",
    "LeaveOneOutCV",
    "ShuffleSplitCV",
    "RepeatedStratifiedKFoldCV",
    "NestedCV",
    # Module 5d
    "ModelComparison",
    # Module 5e
    "VisualizationManager",
    # Module 5f
    "ComprehensiveHyperparameterAnalyzer",
    "SensitivityAnalysis",
    "GeneralizationGapAnalysis",
    "ParameterImportance",
    "OptimizationSuggestions",
    "LearningDynamicsAnalysis",
    # Module 5g
    "ComprehensiveReportManager",
]