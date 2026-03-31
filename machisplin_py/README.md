# MACHISPLIN (Python Version)

MACHISPLIN is an interpolation tool for noisy multivariate data using machine learning ensembling and thin-plate-smoothing splines. It serves as a free, open-source alternative to the commercial ANUSPLIN software.

## Features
- Ensemble machine learning interpolation using up to six algorithms:
  - Boosted Regression Trees (BRT)
  - Random Forests (RF)
  - Neural Networks (NN)
  - Generalized Additive Models (GAM)
  - Multivariate Adaptive Regression Splines (MARS)
  - Support Vector Machines (SVM)
- K-fold cross-validation for optimal model selection and weighting.
- Thin-plate-smoothing spline interpolation of residuals for error correction.
- Tiling capability for processing large landscapes.

## Requirements
- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- rasterio
- pygam
- xgboost
- statsmodels
- pyearth (optional for MARS)

## Installation
You can install the package using:
```bash
cd machisplin_py
pip install .
```

## Example Usage
```python
import pandas as pd
import machisplin

# Load sampling data
my_data = pd.read_csv("sampling.csv") # columns: long, lat, value1, value2, ...

# Path to high-resolution covariates raster (multi-band TIFF)
covar_ras_path = "covariates.tif"

# Run the interpolation
results = machisplin.mltps(int_values=my_data, covar_ras_path=covar_ras_path, tps=True)

# Save the results
machisplin.write_geotiff(results)
machisplin.write_loadings(results)
machisplin.write_residuals(results)
```

## Credits
This is a Python port of the MACHISPLIN R package. Original authors: S. Kreiner et al.
