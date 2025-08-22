import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def open_image_safely(path: Path, max_side: int = 2400) -> Image.Image:
    image = Image.open(path)
    image.thumbnail((max_side, max_side))
    return image

st.set_page_config(page_title="Tehran Weather – Machine Learning Dashboard", layout="wide")
ARTIFACTS_DIRECTORY = Path("artifacts")

st.title("Tehran Weather – Machine Learning Dashboard")

st.header("Data")

merged_path = ARTIFACTS_DIRECTORY / "merged_data_with_additional_features.csv"
if merged_path.exists():
    df_merged = pd.read_csv(merged_path, low_memory=False)
    index_like_cols = [c for c in df_merged.columns if c.lower().startswith("unnamed")]
    if index_like_cols:
        df_merged = df_merged.drop(columns=index_like_cols)

    date_column_name = "date" if "date" in df_merged.columns else None
    if date_column_name:
        df_merged[date_column_name] = pd.to_datetime(df_merged[date_column_name], errors="coerce")

    with st.expander("Preview of merged data (first 50 rows)"):
        st.dataframe(df_merged.head(50), use_container_width=True)

    if date_column_name and pd.api.types.is_datetime64_any_dtype(df_merged[date_column_name]):
        st.subheader("Quick line chart")
        numeric_columns = [
            c for c in df_merged.columns
            if c != date_column_name and pd.api.types.is_numeric_dtype(df_merged[c])
        ]
        if numeric_columns:
            selected_numeric = st.selectbox("Select a numeric column", numeric_columns, index=0)
            st.line_chart(df_merged.set_index(date_column_name)[selected_numeric])
else:
    st.info("Merged dataset not found. Please run the notebook export cell to create the artifacts directory and files.")

st.header("Correlations with tmax_tehran")

correlation_path = ARTIFACTS_DIRECTORY / "correlation_with_tmax_of_Tehran.csv"
if correlation_path.exists():
    df_correlation = pd.read_csv(correlation_path, index_col=0)
    if df_correlation.shape[1] >= 1:
        correlation_series = df_correlation.iloc[:, 0]
        st.subheader("Correlation table")
        st.dataframe(
            correlation_series.to_frame("correlation_with_tmax").sort_values("correlation_with_tmax", ascending=False),
            use_container_width=True
        )

        top5_path = ARTIFACTS_DIRECTORY / "top5_features.json"
        if top5_path.exists():
            top5 = json.loads(top5_path.read_text(encoding="utf-8"))
            st.caption(f"Top five features: {', '.join(top5)}")
    else:
        st.warning("The correlation file exists but has no columns.")
else:
    st.info("Correlation file not found.")

st.header("Key Performance Indicators")
kpi_columns = st.columns(3)

basic_metrics_path = ARTIFACTS_DIRECTORY / "metrics_basic.json"
if basic_metrics_path.exists():
    metrics_basic = json.loads(basic_metrics_path.read_text(encoding="utf-8"))
    mean_value = metrics_basic.get("tmax_tehran_mean")
    std_value = metrics_basic.get("tmax_tehran_std")
    kpi_columns[0].metric("tmax_tehran mean", f"{float(mean_value):.3f}" if mean_value is not None else "—")
    kpi_columns[1].metric("tmax_tehran standard deviation", f"{float(std_value):.3f}" if std_value is not None else "—")
else:
    kpi_columns[0].caption("metrics_basic.json not found.")

best_summary_path = ARTIFACTS_DIRECTORY / "best_model_summary.json"
if best_summary_path.exists():
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    kpi_columns[2].metric("Best model", best_summary.get("best_model", "—"))
    mae_value = best_summary.get("best_MAE")
    r2_value = best_summary.get("best_R2")
    custom_accuracy_value = best_summary.get("best_Accuracy")
    st.write(
        f"**Mean Absolute Error:** {float(mae_value):.4f} | "
        f"**Coefficient of Determination (R²):** {float(r2_value):.4f} | "
        f"**Custom accuracy:** {float(custom_accuracy_value):.4f}"
        if None not in (mae_value, r2_value, custom_accuracy_value) else
        f"**Mean Absolute Error:** {mae_value or '—'} | "
        f"**Coefficient of Determination (R²):** {r2_value or '—'} | "
        f"**Custom accuracy:** {custom_accuracy_value or '—'}"
    )
else:
    kpi_columns[2].caption("best_model_summary.json not found.")

st.header("Model performance and timing")

performance_csv = ARTIFACTS_DIRECTORY / "model_performance.csv"
timing_csv = ARTIFACTS_DIRECTORY / "model_times.csv"

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Performance of models")
    if performance_csv.exists():
        st.dataframe(pd.read_csv(performance_csv), use_container_width=True)
    else:
        st.caption("model_performance.csv not found.")

with right_col:
    st.subheader("Training time and prediction time")
    if timing_csv.exists():
        st.dataframe(pd.read_csv(timing_csv), use_container_width=True)
    else:
        st.caption("model_times.csv not found.")

st.header("Figures")

def show_image(name: str, title: str):
    path = ARTIFACTS_DIRECTORY / name
    if not path.exists():
        st.caption(f"{name} not found.")
        return

    st.subheader(title)
    try:
        try:
            st.image(str(path), use_container_width=True)
        except TypeError:
            st.image(str(path), use_column_width=True)
    except Exception:
        try:
            image = open_image_safely(path, max_side=2400)
            try:
                st.image(image, use_container_width=True)
            except TypeError:
                st.image(image, use_column_width=True)
        except Exception as error:
            st.error(f"Could not display {name}: {error}")

show_image("tmax_histograms.png", "Histograms and kernel density estimate for tmax_tehran")
show_image("pairplot_top5.png", "Pair plot of five top correlated features")
show_image("pairplot_significant.png", "Pair plot of features with significant correlation")
show_image("learning_curve.png", "Learning curve for the best model")
show_image("gb_iterations_mse.png", "Mean squared error versus iterations for Gradient Boosting")
show_image("actual_vs_pred.png", "Actual values versus predicted values for the best model")

st.header("Predictions (best model)")

predictions_path = ARTIFACTS_DIRECTORY / "Predicted_main.csv"
if predictions_path.exists():
    df_predictions = pd.read_csv(predictions_path)
    st.dataframe(df_predictions.head(100), use_container_width=True)
else:
    st.caption("Predicted_main.csv not found.")

st.header("Downloads")

all_files = sorted(ARTIFACTS_DIRECTORY.glob("*"))
if all_files:
    for file_path in all_files:
        file_bytes = file_path.read_bytes()
        st.download_button(
            label=f"Download {file_path.name}",
            data=file_bytes,
            file_name=file_path.name,
            mime="application/octet-stream",
            use_container_width=True
        )
else:
    st.caption("No files found in the artifacts directory.")
