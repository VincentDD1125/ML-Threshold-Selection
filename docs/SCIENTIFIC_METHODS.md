# Scientific Methods and Algorithms

This document provides detailed explanations of the scientific methods and algorithms used in the ML Threshold Selection toolkit.

## Table of Contents

- [Ellipsoid Feature Engineering](#ellipsoid-feature-engineering)
- [Fabric Analysis (T and P' Parameters)](#fabric-analysis-t-and-p-parameters)
- [Bootstrap Analysis](#bootstrap-analysis)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Probability Calculation](#probability-calculation)
- [Threshold Determination](#threshold-determination)

## Ellipsoid Feature Engineering

### Mathematical Foundation

Each particle is represented as an ellipsoid defined by its principal axes and eigenvalues. The ellipsoid tensor is constructed as:

```
T = [λ₁ 0  0 ]
    [0  λ₂ 0 ]
    [0  0  λ₃]
```

Where λ₁, λ₂, λ₃ are the eigenvalues (principal values) of the particle.

### Log-Euclidean Mapping

To enable linear operations on tensors, we map them to the log-Euclidean space:

```
log(T) = [log(λ₁) 0       0     ]
         [0       log(λ₂) 0     ]
         [0       0       log(λ₃)]
```

### 7D Feature Vector

The following 7 features are extracted from each particle:

1. **Volume**: `V = (4π/3) * √(λ₁λ₂λ₃)`
2. **Aspect Ratio 1**: `AR₁ = λ₁/λ₂`
3. **Aspect Ratio 2**: `AR₂ = λ₂/λ₃`
4. **Sphericity**: `S = λ₃/λ₁`
5. **Oblateness**: `O = (λ₁ - λ₂)/λ₁`
6. **Prolateness**: `P = (λ₂ - λ₃)/λ₂`
7. **Shape Factor**: `SF = (λ₁λ₂λ₃)^(1/3) / λ₁`

### Resolution-Aware Normalization

Features are normalized by voxel size to ensure cross-sample compatibility:

```
normalized_feature = feature / (voxel_size_mm)³
```

## Fabric Analysis (T and P' Parameters)

### Jelínek (1981) Methodology

The fabric analysis follows the Jelínek (1981) approach for characterizing particle orientation distributions.

### Log-Euclidean Mean Tensor

For each volume threshold, the mean fabric tensor is computed using log-Euclidean averaging:

```
T_mean = exp(1/n * Σᵢ log(Tᵢ))
```

Where n is the number of particles and Tᵢ is the tensor of particle i.

### Eigenvalue Analysis

The principal values of the mean tensor are extracted:

```
T_mean = [λ₁ 0  0 ]
         [0  λ₂ 0 ]
         [0  0  λ₃]
```

### T and P' Parameters

The fabric parameters are calculated following Jelínek (1981):

**T Parameter**:
```
T = (2f₂ - f₁ - f₃) / (f₁ - f₃)
```

**P' Parameter**:
```
P' = exp√2[(f₁ - f)² + (f₂ - f)² + (f₃ - f)²]
```

Where:
- f₁, f₂, f₃ are the natural logs of the normalized magnitudes of the maximum (Φ₁), intermediate (Φ₂), and minimum (Φ₃) axes of the fabric ellipsoid
- f = (f₁ + f₂ + f₃)/3

For detailed interpretation of these parameters, refer to the original literature (Jelínek, 1981).

## Bootstrap Analysis

### Statistical Resampling

For each volume threshold, particles are resampled with replacement to estimate confidence intervals.

### Algorithm

1. **Bootstrap Sampling**: For each threshold, create B bootstrap samples (default B=1000)
2. **Tensor Computation**: Compute mean tensor for each bootstrap sample
3. **Parameter Calculation**: Calculate T and P' for each bootstrap sample
4. **Confidence Intervals**: Compute 95% confidence intervals from bootstrap distribution

### Confidence Interval Calculation

```
CI_95 = [percentile_2.5, percentile_97.5]
```

## Machine Learning Pipeline

### Feature Standardization

All features are standardized using StandardScaler:

```
X_scaled = (X - μ) / σ
```

Where μ is the mean and σ is the standard deviation.

### Model Training

The system supports multiple algorithms:

- **LightGBM** (default): Gradient boosting with high performance
- **Random Forest**: Ensemble method with good interpretability
- **SVM**: Support vector machine for non-linear boundaries

### Cross-Validation

K-fold cross-validation is used for robust performance estimation:

```
CV_score = 1/k * Σᵢ score(fold_i)
```

### Hyperparameter Optimization

Grid search or random search is used to optimize hyperparameters:

```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
```

## Probability Calculation

### Artifact Probability

The model outputs the probability that a particle is an artifact:

```
P(artifact) = sigmoid(Σᵢ wᵢ * fᵢ + b)
```

Where wᵢ are learned weights, fᵢ are features, and b is the bias term.

### Threshold-Based Probability

For semi-supervised learning, probabilities are derived from expert thresholds:

```
P(artifact) = 1 - exp(-α * (volume - threshold))
```

Where α is a scaling parameter.

## Threshold Determination

### Loose Threshold (Inflection Point)

The loose threshold is identified as the inflection point of the artifact rate curve:

```
d²(artifact_rate)/d(volume)² = 0
```

This represents the optimal balance between artifact removal and particle retention.

### Strict Threshold (Zero Artifacts)

The strict threshold is determined as the volume where all particles have artifact probability > P_threshold:

```
strict_threshold = max(volume | P(artifact) > P_threshold)
```

Default P_threshold = 0.01 (configurable).

### Grid Search

A volume grid is generated for threshold evaluation:

```python
v_min = min(volumes) * 0.1
v_max = max(volumes) * 10
volume_grid = np.logspace(log10(v_min), log10(v_max), n_points)
```

### Optimization Criteria

The optimal threshold minimizes the combined cost:

```
Cost = α * artifact_rate + β * particle_loss_rate
```

Where α and β are weighting factors.

## Implementation Details

### Memory Optimization

- Use `float32` for large datasets
- Process samples in batches
- Enable data type optimization

### Performance Optimization

- Parallel processing where available
- Feature selection for high-dimensional data
- Efficient tensor operations using NumPy

### Numerical Stability

- Add small epsilon values to prevent division by zero
- Use log-sum-exp trick for numerical stability
- Regularize eigenvalues to prevent singular matrices

## References

### Core Fabric Analysis Methods

1. **Jelínek, V. (1981)**. Characterization of the magnetic fabric of rocks. *Tectonophysics*, 79(1-4), T63-T67. DOI: 10.1016/0040-1951(81)90110-4

2. **Jelínek, V. (1978)**. Statistical processing of anisotropy of magnetic susceptibility measured on groups of specimens. *Studia Geophysica et Geodaetica*, 22(1), 50-62. DOI: 10.1007/BF01613632

### Stereographic Projection and TomoFab

3. **Ketcham, R. A. (2005)**. Three-dimensional grain fabric measurements using high-resolution X-ray computed tomography. *Journal of Structural Geology*, 27(7), 1217-1228. DOI: 10.1016/j.jsg.2005.04.001

4. **Ketcham, R. A., & Ryan, T. M. (2004)**. Quantification and visualization of anisotropy in trabecular bone. *Journal of Microscopy*, 213(2), 158-171. DOI: 10.1111/j.1365-2818.2004.01277.x

5. **TomoFab Software**: https://github.com/ctlab/TomoFab (Accessed: 2024)

### Log-Euclidean Tensor Methods

6. **Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2006)**. Log-Euclidean metrics for fast and simple calculus on diffusion tensors. *Magnetic Resonance in Medicine*, 56(2), 411-421. DOI: 10.1002/mrm.20965

7. **Pennec, X., Fillard, P., & Ayache, N. (2006)**. A Riemannian framework for tensor computing. *International Journal of Computer Vision*, 66(1), 41-66. DOI: 10.1007/s11263-005-3222-z

### Bootstrap and Statistical Methods

8. **Efron, B., & Tibshirani, R. J. (1994)**. *An introduction to the bootstrap*. CRC press. ISBN: 978-0-412-04231-7

9. **Chernick, M. R. (2008)**. *Bootstrap methods: a guide for practitioners and researchers*. John Wiley & Sons. ISBN: 978-0-470-11429-0

### Machine Learning in Geosciences

10. **Hall, B. (2016)**. Facies classification using machine learning. *The Leading Edge*, 35(10), 906-909. DOI: 10.1190/tle35100906.1

11. **Bergen, K. J., Johnson, P. A., de Hoop, M. V., & Beroza, G. C. (2019)**. Machine learning for data-driven discovery in solid Earth geoscience. *Science*, 363(6433), eaau0323. DOI: 10.1126/science.aau0323

### X-ray Computed Tomography

12. **Ketcham, R. A., & Carlson, W. D. (2001)**. Acquisition, optimization and interpretation of X-ray computed tomographic imagery: applications to the geosciences. *Computers & Geosciences*, 27(4), 381-400. DOI: 10.1016/S0098-3004(00)00116-3

13. **Cnudde, V., & Boone, M. N. (2013)**. High-resolution X-ray computed tomography in geosciences: A review of the current technology and applications. *Earth-Science Reviews*, 123, 1-17. DOI: 10.1016/j.earscirev.2013.04.003

### Particle Analysis and Fabric

14. **Passchier, C. W., & Trouw, R. A. J. (2005)**. *Microtectonics*. Springer Science & Business Media. ISBN: 978-3-540-64003-5

15. **Ramsay, J. G., & Huber, M. I. (1987)**. *The techniques of modern structural geology: Volume 2: Folds and fractures*. Academic Press. ISBN: 978-0-12-576922-0

### Software and Tools

16. **MTEX Toolbox**: https://mtex-toolbox.github.io/ (Accessed: 2024)

17. **OIM Analysis Software**: https://www.edax.com/products/ebsd-oim-analysis (Accessed: 2024)

18. **Drishti Software**: https://github.com/nci/drishti (Accessed: 2024)

### Additional Reading

19. **Borradaile, G. J., & Jackson, M. (2010)**. Structural geology, petrofabrics and magnetic fabrics (AMS, AARM, AIRM). *Geological Society, London, Special Publications*, 238(1), 1-14. DOI: 10.1144/GSL.SP.2004.238.01.01

20. **Tarling, D. H., & Hrouda, F. (1993)**. *The magnetic anisotropy of rocks*. Chapman & Hall. ISBN: 978-0-412-40240-5
