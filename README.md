# Retail Demand Forecasting and Inventory Optimization

## Project Overview
This project combines time-series forecasting and machine learning to predict daily retail product demand and support inventory planning.

## Problem Statement
Retail businesses need accurate demand forecasts to reduce stockouts, avoid excess inventory, and improve replenishment decisions.

## Dataset
- **Records:** 4,565,000
- **Stores:** 50
- **Products:** 50
- **Period:** January 2019 – December 2023
- **Granularity:** Daily store-product sales

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Statsmodels (SARIMA/SARIMAX)
- XGBoost
- Scikit-learn
- Google Colab

## Models and Results

### XGBoost
For Store 12, Item 20:
- MAE: 2.81
- RMSE: 3.41
- WAPE: 5.52%

### SARIMA
For Store 12, Item 20, over the 2023 test period:
- MAE: 12.57
- RMSE: 15.78
- WAPE: 20.24%

**Note:** These results use different forecasting evaluation setups, so they should not be treated as a strict head-to-head comparison.

## Inventory Optimization
The project generates inventory recommendations for store-product combinations using forecast demand, lead time, and safety stock assumptions.

## Project Files
- `retail_demand_forecasting.ipynb` — analysis and modeling notebook
- `retail_model_comparison.csv` — model comparison metrics
- `inventory_recommendations (1).csv` — inventory recommendations
- `model_performance_comparison.png` — model performance visualization
- `sarima_forecast_comparison.png` — SARIMA forecast comparison
- `top_10_inventory_recommendations.png` — top inventory recommendations

## Conclusion
This project demonstrates how time-series forecasting and machine learning can support retail demand prediction and data-driven inventory planning.

## Author
Yashika



