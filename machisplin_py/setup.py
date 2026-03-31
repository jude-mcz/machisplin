from setuptools import setup, find_packages

setup(
    name="machisplin",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "rasterio",
        "pygam",
        "matplotlib",
        "xgboost",
        "statsmodels"
    ],
    author="MACHISPLIN Authors",
    description="Python port of MACHISPLIN R package for interpolation of noisy multivariate data",
    url="https://github.com/skreiner/MACHISPLIN",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
