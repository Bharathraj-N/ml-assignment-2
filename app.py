import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}


@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    with open(model_path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_saved_metrics():
    metrics_path = os.path.join(MODEL_DIR, 'metrics.pkl')
    with open(metrics_path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_names():
    path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'AUC': round(roc_auc_score(y_true, y_proba), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1 Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Good (0)', 'Good (1)'],
                yticklabels=['Not Good (0)', 'Good (1)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    return fig


def main():
    st.title("Wine Quality Classification")
    st.markdown("""
    This application demonstrates **binary classification** on the 
    [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).
    Wine samples are classified as **Good** (quality >= 7) or **Not Good** (quality < 7)
    using 5 different ML models.
    """)

    st.sidebar.header("Configuration")

    model_name = st.sidebar.selectbox(
        "Select Classification Model",
        list(MODEL_FILES.keys())
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (test data)",
        type=['csv'],
        help="Upload a CSV with the same features as the wine quality dataset"
    )

    feature_names = load_feature_names()

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head(10), width='stretch')

        if 'target' not in data.columns:
            st.error("CSV must contain a 'target' column for evaluation.")
            return

        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            st.error(f"Missing features in uploaded data: {missing_features}")
            return

        X = data[feature_names]
        y = data['target']

        scaler = load_scaler()
        X_scaled = scaler.transform(X)

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs([
            "Single Model Results", "All Models Comparison", "Classification Report"
        ])

        with tab1:
            st.subheader(f"Results: {model_name}")
            model = load_model(model_name)
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]

            metrics = compute_metrics(y, y_pred, y_proba)

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
            col3.metric("Precision", f"{metrics['Precision']:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{metrics['Recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            col6.metric("MCC", f"{metrics['MCC']:.4f}")

            st.markdown("#### Confusion Matrix")
            fig = plot_confusion_matrix(y, y_pred, model_name)
            st.pyplot(fig)

        with tab2:
            st.subheader("Comparison of All Models")
            all_results = {}
            for name in MODEL_FILES.keys():
                m = load_model(name)
                pred = m.predict(X_scaled)
                proba = m.predict_proba(X_scaled)[:, 1]
                all_results[name] = compute_metrics(y, pred, proba)

            comparison_df = pd.DataFrame(all_results).T
            comparison_df.index.name = 'Model'
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                width='stretch'
            )

            st.markdown("#### Accuracy Comparison")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            models_list = list(all_results.keys())
            accuracies = [all_results[m]['Accuracy'] for m in models_list]
            colors = sns.color_palette("viridis", len(models_list))
            bars = ax2.bar(models_list, accuracies, color=colors)
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Model Accuracy Comparison')
            ax2.set_ylim(0, 1)
            for bar, acc in zip(bars, accuracies):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig2)

        with tab3:
            st.subheader(f"Classification Report - {model_name}")
            model = load_model(model_name)
            y_pred = model.predict(X_scaled)
            report = classification_report(y, y_pred, target_names=['Not Good', 'Good'])
            st.code(report)

    else:
        st.info("Please upload a test CSV file using the sidebar to evaluate models.")
        st.markdown("---")

        st.subheader("Pre-computed Results (Training Phase)")
        st.markdown("Below are the metrics from the training/testing phase:")

        saved_metrics = load_saved_metrics()
        comparison_df = pd.DataFrame(saved_metrics).T
        comparison_df.index.name = 'Model'
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen'),
            width='stretch'
        )

        st.markdown("---")
        st.markdown("### Dataset Information")
        st.markdown("""
        - **Dataset**: UCI Wine Quality (Red + White combined)
        - **Instances**: 6,497
        - **Features**: 12 (11 physicochemical properties + wine type)
        - **Target**: Binary (Good: quality >= 7, Not Good: quality < 7)
        - **Test Split**: 20% (1,300 samples)
        """)

        st.markdown("### Feature Description")
        features_info = pd.DataFrame({
            'Feature': ['fixed acidity', 'volatile acidity', 'citric acid',
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates',
                       'alcohol', 'wine_type'],
            'Description': [
                'Tartaric acid concentration (g/dm³)',
                'Acetic acid concentration (g/dm³)',
                'Citric acid concentration (g/dm³)',
                'Remaining sugar after fermentation (g/dm³)',
                'Sodium chloride concentration (g/dm³)',
                'Free form of SO₂ (mg/dm³)',
                'Total SO₂ (mg/dm³)',
                'Density of wine (g/cm³)',
                'pH level (0-14 scale)',
                'Potassium sulphate concentration (g/dm³)',
                'Alcohol content (% vol)',
                'Type of wine (0=Red, 1=White)'
            ]
        })
        st.table(features_info)


if __name__ == '__main__':
    main()
