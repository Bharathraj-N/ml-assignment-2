import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

DATA_URL_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def load_wine_quality_data():
    """Load and combine red and white wine quality datasets from UCI."""
    red_wine = pd.read_csv(DATA_URL_RED, sep=';')
    white_wine = pd.read_csv(DATA_URL_WHITE, sep=';')

    red_wine['wine_type'] = 0
    white_wine['wine_type'] = 1

    data = pd.concat([red_wine, white_wine], ignore_index=True)

    # Binary classification: quality >= 7 is "good" (1), else "not good" (0)
    data['target'] = (data['quality'] >= 7).astype(int)
    data.drop('quality', axis=1, inplace=True)

    return data


def train_and_evaluate():
    """Train all 5 models and save them along with metrics."""
    print("Loading Wine Quality dataset...")
    data = load_wine_quality_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()[:-1]}")
    print(f"Target distribution:\n{data['target'].value_counts()}")

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save test data
    test_data = X_test.copy()
    test_data['target'] = y_test.values
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    test_data.to_csv(os.path.join(project_dir, 'test_data.csv'), index=False)
    print(f"\nTest data saved: {X_test.shape[0]} samples")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_proba), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        results[name] = metrics

        model_path = os.path.join(script_dir, f"{name.lower().replace(' ', '_').replace('-', '_')}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"  Accuracy: {metrics['Accuracy']}")
        print(f"  AUC: {metrics['AUC']}")
        print(f"  Precision: {metrics['Precision']}")
        print(f"  Recall: {metrics['Recall']}")
        print(f"  F1: {metrics['F1']}")
        print(f"  MCC: {metrics['MCC']}")

    # Save scaler
    with open(os.path.join(script_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save results
    with open(os.path.join(script_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Save feature names
    with open(os.path.join(script_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    print("="*70)

    return results


if __name__ == '__main__':
    train_and_evaluate()
