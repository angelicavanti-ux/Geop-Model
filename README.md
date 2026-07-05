# Geop-Model

## Project

This repository contains the code, data and documentation for the dissertation project:

**Predicting Cross-Currency Basis Spread Volatility Using Geopolitical and Macroeconomic Variables**

## Repository Structure

```
data/
    raw/           # Original datasets imported from external sources
    processed/     # Cleaned and transformed datasets

figures/           # Figures generated throughout the project

notebooks/         # Jupyter notebooks used for data collection, exploration and modelling

src/               # Python scripts and reusable functions
```

## Current Progress

- Imported the four macroeconomic control variables from FRED.
- Explored and visualised each dataset.
- Performed Augmented Dickey-Fuller (ADF) tests.
- Applied transformations to achieve stationarity where required.

## Next Steps

- Obtain Cross-Currency Basis (CCB) data.
- Construct the VAR-X dataset.
- Estimate and evaluate the forecasting model.