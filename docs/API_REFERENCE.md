# API Reference

Complete API documentation for the ML Threshold Selection toolkit.

## Table of Contents

- [Core Classes](#core-classes)
- [Feature Engineering](#feature-engineering)
- [Fabric Analysis](#fabric-analysis)
- [Utility Functions](#utility-functions)
- [Data I/O](#data-io)
- [Visualization](#visualization)

## Core Classes

### SupervisedThresholdLearner

Main class for supervised learning approach.

```python
class SupervisedThresholdLearner:
    def __init__(self, model_type='lightgbm', random_state=42):
        """
        Initialize supervised threshold learner.
        
        Parameters:
        -----------
        model_type : str, default='lightgbm'
            Type of machine learning model to use
        random_state : int, default=42
            Random state for reproducibility
        """
    
    def train(self, features, labels, model_type='lightgbm', cv_folds=5):
        """
        Train the supervised learning model.
        
        Parameters:
        -----------
        features : array-like, shape (n_samples, n_features)
            Feature matrix
        labels : array-like, shape (n_samples,)
            Binary labels (0=particle, 1=artifact)
        model_type : str, default='lightgbm'
            Type of model to train
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Training results including CV scores
        """
    
    def analyze_sample(self, data, return_features=False):
        """
        Analyze a sample and predict optimal thresholds.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data with required columns
        return_features : bool, default=False
            Whether to return extracted features
            
        Returns:
        --------
        dict : Analysis results including thresholds and statistics
        """
    
    def predict_proba(self, features):
        """
        Predict artifact probabilities for given features.
        
        Parameters:
        -----------
        features : array-like, shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        array : Artifact probabilities
        """
    
    def find_dual_thresholds(self, volumes, probabilities, strict_probability_threshold=0.01):
        """
        Find loose and strict thresholds from probabilities.
        
        Parameters:
        -----------
        volumes : array-like
            Particle volumes
        probabilities : array-like
            Artifact probabilities
        strict_probability_threshold : float, default=0.01
            Probability threshold for strict threshold
            
        Returns:
        --------
        dict : Threshold results
        """
```

### SemiSupervisedThresholdLearner

Main class for semi-supervised learning approach.

```python
class SemiSupervisedThresholdLearner:
    def __init__(self, model_type='lightgbm', random_state=42):
        """
        Initialize semi-supervised threshold learner.
        
        Parameters:
        -----------
        model_type : str, default='lightgbm'
            Type of machine learning model to use
        random_state : int, default=42
            Random state for reproducibility
        """
    
    def add_expert_threshold(self, sample_id, threshold, confidence=1.0):
        """
        Add expert threshold for a sample.
        
        Parameters:
        -----------
        sample_id : str
            Sample identifier
        threshold : float
            Expert-determined threshold
        confidence : float, default=1.0
            Confidence in the threshold (0-1)
        """
    
    def load_sample_data(self, sample_id, file_path):
        """
        Load particle data for a sample.
        
        Parameters:
        -----------
        sample_id : str
            Sample identifier
        file_path : str
            Path to data file
        """
    
    def train(self, method='threshold_based', model_type='lightgbm'):
        """
        Train the semi-supervised model.
        
        Parameters:
        -----------
        method : str, default='threshold_based'
            Training method ('threshold_based' or 'pseudo_labeling')
        model_type : str, default='lightgbm'
            Type of model to train
            
        Returns:
        --------
        dict : Training results
        """
    
    def analyze_sample(self, data, return_features=False):
        """
        Analyze a sample and predict optimal thresholds.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data with required columns
        return_features : bool, default=False
            Whether to return extracted features
            
        Returns:
        --------
        dict : Analysis results including thresholds and statistics
        """
```

## Feature Engineering

### FeatureEngineer

Feature extraction and engineering utilities.

```python
class FeatureEngineer:
    def __init__(self, voxel_size_mm=0.03):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        voxel_size_mm : float, default=0.03
            Voxel size in millimeters
        """
    
    def extract_all_features(self, data, voxel_size_mm=None):
        """
        Extract all features from particle data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data with required columns
        voxel_size_mm : float, optional
            Voxel size for normalization
            
        Returns:
        --------
        pd.DataFrame : Feature matrix
        """
    
    def extract_ellipsoid_features(self, data):
        """
        Extract ellipsoid-based features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data with eigenvalue/eigenvector columns
            
        Returns:
        --------
        pd.DataFrame : Ellipsoid features
        """
    
    def extract_resolution_aware_features(self, data, voxel_size_mm):
        """
        Extract resolution-aware features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data
        voxel_size_mm : float
            Voxel size for normalization
            
        Returns:
        --------
        pd.DataFrame : Resolution-aware features
        """
    
    def normalize_features(self, features, voxel_size_mm):
        """
        Normalize features by voxel size.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        voxel_size_mm : float
            Voxel size for normalization
            
        Returns:
        --------
        pd.DataFrame : Normalized features
        """
```

## Fabric Analysis

### FabricAnalyzer

Fabric analysis and bootstrap computation.

```python
class FabricAnalyzer:
    def __init__(self, n_bootstrap=1000, confidence_level=0.95):
        """
        Initialize fabric analyzer.
        
        Parameters:
        -----------
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        confidence_level : float, default=0.95
            Confidence level for intervals
        """
    
    def compute_fabric_parameters(self, data, volume_threshold):
        """
        Compute fabric parameters for a given threshold.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data
        volume_threshold : float
            Volume threshold for filtering
            
        Returns:
        --------
        dict : Fabric parameters (T, P', eigenvalues)
        """
    
    def bootstrap_analysis(self, data, volume_threshold):
        """
        Perform bootstrap analysis for confidence intervals.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Particle data
        volume_threshold : float
            Volume threshold for filtering
            
        Returns:
        --------
        dict : Bootstrap results with confidence intervals
        """
    
    def compute_log_euclidean_mean(self, tensors):
        """
        Compute log-Euclidean mean of tensors.
        
        Parameters:
        -----------
        tensors : array-like, shape (n_particles, 3, 3)
            Particle tensors
            
        Returns:
        --------
        array : Mean tensor
        """
    
    def calculate_t_p_parameters(self, eigenvalues):
        """
        Calculate T and P' parameters from eigenvalues.
        
        Parameters:
        -----------
        eigenvalues : array-like, shape (3,)
            Principal eigenvalues
            
        Returns:
        --------
        tuple : (T, P') parameters
        """
```

## Utility Functions

### Data Validation

```python
def validate_particle_data(data, required_columns=None):
    """
    Validate particle data format and content.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Particle data to validate
    required_columns : list, optional
        List of required columns
        
    Returns:
    --------
    dict : Validation results
    """
```

### Threshold Finding

```python
def find_inflection_threshold(volumes, artifact_rates):
    """
    Find inflection point threshold.
    
    Parameters:
    -----------
    volumes : array-like
        Volume values
    artifact_rates : array-like
        Corresponding artifact rates
        
    Returns:
    --------
    float : Inflection threshold
    """
```

### Probability Calculation

```python
def calculate_artifact_probabilities(features, model):
    """
    Calculate artifact probabilities using trained model.
    
    Parameters:
    -----------
    features : array-like
        Feature matrix
    model : trained model
        Trained machine learning model
        
    Returns:
    --------
    array : Artifact probabilities
    """
```

## Data I/O

### Data Loading

```python
def load_particle_data(file_path, sheet_name=None):
    """
    Load particle data from file.
    
    Parameters:
    -----------
    file_path : str
        Path to data file
    sheet_name : str, optional
        Sheet name for Excel files
        
    Returns:
    --------
    pd.DataFrame : Loaded data
    """
```

### Data Saving

```python
def save_results(results, output_path, format='excel'):
    """
    Save analysis results to file.
    
    Parameters:
    -----------
    results : dict
        Analysis results
    output_path : str
        Output file path
    format : str, default='excel'
        Output format ('excel', 'csv', 'json')
    """
```

## Visualization

### Plotting Functions

```python
def plot_dual_threshold_analysis(volumes, probabilities, loose_threshold, strict_threshold):
    """
    Plot dual threshold analysis.
    
    Parameters:
    -----------
    volumes : array-like
        Volume values
    probabilities : array-like
        Artifact probabilities
    loose_threshold : float
        Loose threshold value
    strict_threshold : float
        Strict threshold value
    """
```

```python
def plot_fabric_boxplots(t_values, p_values, thresholds, sample_id):
    """
    Plot fabric parameter boxplots.
    
    Parameters:
    -----------
    t_values : array-like
        T parameter values
    p_values : array-like
        P' parameter values
    thresholds : array-like
        Volume thresholds
    sample_id : str
        Sample identifier
    """
```

### GUI Components

```python
class ThresholdAnalysisGUI:
    def __init__(self):
        """Initialize the GUI application."""
    
    def create_main_window(self):
        """Create the main application window."""
    
    def setup_layout(self):
        """Setup the GUI layout."""
    
    def load_training_data(self):
        """Load training data from file."""
    
    def input_expert_thresholds(self):
        """Input expert thresholds dialog."""
    
    def train_model(self):
        """Train the machine learning model."""
    
    def analyze_sample(self):
        """Analyze a test sample."""
    
    def export_results(self):
        """Export analysis results."""
```

## Configuration

### Configuration Parameters

```python
# config.py
STRICT_PROBABILITY_THRESHOLD = 0.01
DEFAULT_VOXEL_SIZE_MM = 0.03
N_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
MODEL_TYPE = 'lightgbm'
RANDOM_STATE = 42
```

## Error Handling

### Custom Exceptions

```python
class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

class ThresholdAnalysisError(Exception):
    """Raised when threshold analysis fails."""
    pass
```

## Performance Optimization

### Memory Management

```python
def optimize_memory_usage(data):
    """
    Optimize memory usage for large datasets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
        
    Returns:
    --------
    pd.DataFrame : Memory-optimized data
    """
```

### Parallel Processing

```python
def parallel_bootstrap_analysis(data, n_workers=4):
    """
    Perform bootstrap analysis in parallel.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Particle data
    n_workers : int, default=4
        Number of parallel workers
        
    Returns:
    --------
    dict : Bootstrap results
    """
```
