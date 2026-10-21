
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

st.title("Retail Demand Forecasting & Inventory Optimization")
st.caption("Machine Learning | Time-Series Forecasting | Inventory Planning")

# File paths
model_path = BASE_DIR / "retail_model_comparison.csv"
inventory_path = BASE_DIR / "inventory_recommendations.csv"
forecast_image = BASE_DIR / "sarima_forecast_comparison.png"
performance_image = BASE_DIR / "model_performance_comparison.png"
predictions_path = BASE_DIR / "xgboost_test_predictions.csv"

# Sidebar navigation
st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Model Comparison",
        "XGBoost Forecast",
        "Inventory Recommendations"
    ]
)

# --------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------
if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This project uses machine learning and time-series "
        "analysis to forecast retail demand and support inventory planning."
    )

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "4.56 Million")
    col2.metric("Stores Analysed", "50")
    col3.metric("Products Analysed", "50")

    col4, col5, col6 = st.columns(3)
    col4.metric("Store-Product Combinations", "2,500")
    col5.metric("Data Period", "2019–2023")
    col6.metric("Forecast Model", "XGBoost")

    st.subheader("XGBoost Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", "2.52")
    col2.metric("RMSE", "3.16")
    col3.metric("WAPE", "9.52%")

    st.caption(
        "Metrics are from the existing Google Colab evaluation."
    )

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyse historical retail sales data.
    - Forecast future product demand using machine learning.
    - Compare forecasting model results.
    - Calculate inventory reorder points.
    - Support better inventory planning and reduce stockout risk.
    """)

    st.info(
        "This dashboard presents results from the existing project."
    )

    st.subheader("SARIMA Forecast Visualization")

    if forecast_image.exists():
        st.image(
            str(forecast_image),
            caption="SARIMA Forecast Comparison",
            use_container_width=True
        )
    else:
        st.warning("SARIMA forecast image was not found.")

    st.subheader("Model Performance Comparison")

    if performance_image.exists():
        st.image(
            str(performance_image),
            caption="Forecast Model Performance Comparison",
            use_container_width=True
        )
    else:
        st.warning("Model performance comparison image was not found.")

# --------------------------------------------------
# 2. MODEL COMPARISON
# --------------------------------------------------
elif page == "Model Comparison":

    st.header("Forecast Model Comparison")

    if model_path.exists():

        models = pd.read_csv(model_path)

        st.subheader("Model Results Table")
        st.dataframe(models, use_container_width=True)

        numeric_columns = models.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            metric = st.selectbox(
                "Select a metric to visualise",
                numeric_columns
            )

            chart_data = models[
                [models.columns[0], metric]
            ].dropna()

            fig = px.bar(
                chart_data,
                x=models.columns[0],
                y=metric,
                title=f"Model Comparison: {metric}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.warning("No numeric metrics were found in the CSV.")

    else:
        st.error("Cannot find retail_model_comparison.csv.")

# --------------------------------------------------
# 3. XGBOOST FORECAST
# --------------------------------------------------
elif page == "XGBoost Forecast":

    st.header("XGBoost Actual vs Predicted Sales")

    if predictions_path.exists():

        predictions = pd.read_csv(predictions_path)

        st.subheader("Prediction Data Preview")
        st.dataframe(
            predictions.head(100),
            use_container_width=True
        )

        st.download_button(
            label="Download XGBoost Predictions",
            data=predictions.to_csv(index=False),
            file_name="xgboost_test_predictions.csv",
            mime="text/csv"
        )

        # Identify likely actual and predicted sales columns
        actual_candidates = [
            "actual", "actual_sales", "y_test",
            "sales", "y_true", "actual_value"
        ]

        predicted_candidates = [
            "predicted", "predicted_sales", "y_pred",
            "prediction", "xgboost_prediction",
            "predicted_value"
        ]

        lower_columns = {
            str(column).strip().lower(): column
            for column in predictions.columns
        }

        actual_column = next(
            (
                lower_columns[name]
                for name in actual_candidates
                if name in lower_columns
            ),
            None
        )

        predicted_column = next(
            (
                lower_columns[name]
                for name in predicted_candidates
                if name in lower_columns
            ),
            None
        )

        # Match columns by common naming patterns if needed
        if actual_column is None:
            actual_column = next(
                (
                    column for column in predictions.columns
                    if "actual" in str(column).lower()
                    or "y_true" in str(column).lower()
                ),
                None
            )

        if predicted_column is None:
            predicted_column = next(
                (
                    column for column in predictions.columns
                    if "predict" in str(column).lower()
                    or "y_pred" in str(column).lower()
                ),
                None
            )

        if actual_column and predicted_column:

            chart_data = predictions.copy()

            # Show a sample for a clearer chart on large files
            if len(chart_data) > 500:
                chart_data = chart_data.tail(500)

            chart_data = chart_data.reset_index(drop=True)
            chart_data["Test Row"] = range(1, len(chart_data) + 1)

            plot_data = chart_data[
                [actual_column, predicted_column, "Test Row"]
            ].melt(
                id_vars="Test Row",
                value_vars=[actual_column, predicted_column],
                var_name="Sales Type",
                value_name="Sales"
            )

            fig = px.line(
                plot_data,
                x="Test Row",
                y="Sales",
                color="Sales Type",
                title="Actual vs Predicted Sales",
                labels={
                    "Test Row": "Test Observation",
                    "Sales": "Sales"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "The chart displays up to the last 500 observations "
                "from the prediction file."
            )

        else:
            st.warning(
                "The actual and predicted sales columns could not be "
                "identified automatically. Check the column names in "
                "the preview table above."
            )
            st.write("Available columns:", list(predictions.columns))

    else:
        st.error(
            "Cannot find xgboost_test_predictions.csv. "
            "Make sure it is in the same folder as app.py."
        )

# --------------------------------------------------
# 4. INVENTORY RECOMMENDATIONS
# --------------------------------------------------
elif page == "Inventory Recommendations":

    st.header("Inventory Recommendations")

    if inventory_path.exists():

        inventory = pd.read_csv(inventory_path)

        col1, col2, col3 = st.columns(3)

        col1.metric("Recommendation Rows", f"{len(inventory):,}")

        if "store_id" in inventory.columns:
            col2.metric(
                "Stores",
                f"{inventory['store_id'].nunique():,}"
            )

        if "item_id" in inventory.columns:
            col3.metric(
                "Products",
                f"{inventory['item_id'].nunique():,}"
            )

        st.subheader("Inventory Data")
        st.dataframe(
            inventory.head(100),
            use_container_width=True
        )

        st.download_button(
            label="Download Inventory CSV",
            data=inventory.to_csv(index=False),
            file_name="inventory_recommendations.csv",
            mime="text/csv"
        )

        if "reorder_point" in inventory.columns:

            st.subheader("Top 10 Reorder Points")

            top_inventory = inventory.nlargest(
                10,
                "reorder_point"
            )

            x_column = (
                "item_id"
                if "item_id" in top_inventory.columns
                else top_inventory.columns[0]
            )

            color_column = (
                "store_id"
                if "store_id" in top_inventory.columns
                else None
            )

            fig = px.bar(
                top_inventory,
                x=x_column,
                y="reorder_point",
                color=color_column,
                title="Top 10 Store-Product Reorder Points",
                labels={
                    "reorder_point": "Reorder Point",
                    "item_id": "Product",
                    "store_id": "Store"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning(
                "The inventory CSV has no reorder_point column."
            )

    else:
        st.error("Cannot find inventory_recommendations.csv.")
