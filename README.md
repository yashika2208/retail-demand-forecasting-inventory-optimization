# Retail Demand Forecasting and Inventory Optimization

## 1. Project Overview
This project uses time-series analysis and machine learning
to forecast retail product demand and support inventory planning.

## 2. Problem Statement
Retail businesses need accurate demand forecasts to reduce
stockouts, avoid excess inventory, and improve stock planning.

## 3. Dataset
- 4,565,000 records
- 50 stores
- 50 products
- Daily sales data from 2019 to 2023
- Features include date, store_id, item_id, sales,
  price, promo, weekday, and month

## 4. Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Statsmodels
- Scikit-learn
- XGBoost
- Google Colab

## 5. Models
- SARIMA
- SARIMAX
- XGBoost Regressor

## 6. Model Evaluation
Store 12, Item 20, July-December 2023:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| SARIMA | 12.14 | 13.96 | 23.82% |
| XGBoost | 2.81 | 3.41 | 5.52% |

These results use different forecasting strategies:
XGBoost uses observed lagged sales during the test period,
while SARIMA produces a multi-step forecast. Therefore,
the comparison is preliminary and not fully like-for-like.

## 7. Inventory Optimization
The project calculates:
- Average predicted daily demand
- Safety stock
- Reorder point
- Target stock level

The current inventory calculations use assumed supplier
lead time and safety-stock percentage. Real inventory data
is needed for operational recommendations.

## 8. Outputs
- XGBoost test predictions
- SARIMA model comparison
- Inventory recommendations
- Forecasting dashboard
- Model performance charts

## 9. Future Improvements
- Perform a fair comparison using the same forecast horizon
- Tune model parameters
- Test additional forecasting models
- Use real supplier lead times and current stock levels
- Build an interactive Power BI dashboard

## 10. Conclusion
The project demonstrates how time-series forecasting and
machine learning can support retail demand prediction and
inventory planning.
Author
Yashika
