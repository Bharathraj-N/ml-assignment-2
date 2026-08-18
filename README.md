# Wine Quality Classification - ML Assignment 2

## Problem Statement

The goal of this project is to build a **binary classification system** that predicts whether a wine sample is of **good quality** (quality rating >= 7) or **not good quality** (quality rating < 7) based on its physicochemical properties. We implement and compare 5 different machine learning classification algorithms and deploy an interactive web application using Streamlit for model evaluation and visualization.

## Dataset Description

- **Dataset**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) (Red + White combined)
- **Source**: UCI Machine Learning Repository
- **Total Instances**: 6,497 (1,599 red + 4,898 white)
- **Features**: 12 input features (11 physicochemical properties + wine type indicator)
- **Target**: Binary classification (Good: quality >= 7 → class 1, Not Good: quality < 7 → class 0)
- **Class Distribution**: 5,220 Not Good (80.3%) | 1,277 Good (19.7%)

### Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | fixed acidity | Tartaric acid concentration (g/dm³) |
| 2 | volatile acidity | Acetic acid concentration (g/dm³) |
| 3 | citric acid | Citric acid concentration (g/dm³) |
| 4 | residual sugar | Sugar remaining after fermentation (g/dm³) |
| 5 | chlorides | Sodium chloride concentration (g/dm³) |
| 6 | free sulfur dioxide | Free form of SO₂ (mg/dm³) |
| 7 | total sulfur dioxide | Total SO₂ bound + free (mg/dm³) |
| 8 | density | Density of wine (g/cm³) |
| 9 | pH | pH level (0-14 scale) |
| 10 | sulphates | Potassium sulphate concentration (g/dm³) |
| 11 | alcohol | Alcohol content (% by volume) |
| 12 | wine_type | Type of wine (0 = Red, 1 = White) |

## GitHub Repository Link

[GitHub Repository](https://github.com/Bharathraj-N/ml-assignment-2)

## Live Streamlit App

[Streamlit App](https://YOUR_APP_URL.streamlit.app)

## Models Used

All models were trained on 80% of the data (5,197 samples) and evaluated on a 20% held-out test set (1,300 samples). Standard scaling was applied to all features before training.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8538 | 0.7749 | 0.6250 | 0.6445 | 0.6346 | 0.5434 |
| K-Nearest Neighbors | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| Naive Bayes | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Random Forest | 0.8923 | 0.9120 | 0.8333 | 0.5664 | 0.6744 | 0.6291 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieves decent accuracy (82.2%) but struggles with recall (26.2%) for the minority "Good" class. The model is overly conservative — it rarely predicts "Good" but when it does, it's moderately reliable (61.5% precision). The linear decision boundary is insufficient to capture the complex non-linear relationships between wine features and quality. |
| Decision Tree | Provides a good balance between precision (62.5%) and recall (64.5%), resulting in a strong F1 score (0.6346). The tree structure captures non-linear feature interactions effectively. However, the lower AUC (0.7749) compared to KNN and Random Forest suggests potential overfitting to training data patterns. |
| K-Nearest Neighbors | Shows moderate performance across all metrics. The AUC of 0.8264 indicates reasonable discriminative ability between classes. However, the relatively low recall (47.7%) means it misses nearly half the "Good" wines. Performance is sensitive to the choice of K and distance metric. |
| Naive Bayes | Has the lowest accuracy (73.5%) and highest false positive rate. While it achieves relatively good recall (61.7%), the precision is poor (39.0%), leading to many false positives. The independence assumption between features is clearly violated for this dataset where features like density, alcohol, and residual sugar are correlated. |
| Random Forest | **Best overall performer** with highest accuracy (89.2%), AUC (0.912), precision (83.3%), F1 (0.6744), and MCC (0.6291). The ensemble of 100 trees effectively handles feature interactions and reduces overfitting compared to a single decision tree. Slightly lower recall (56.6%) indicates some conservative predictions, but the high precision means very few false positives. |
| **Overall Winner** | **Random Forest** is the clear winner for this dataset. It achieves the best scores in 5 out of 6 metrics (Accuracy, AUC, Precision, F1, MCC). Its ensemble nature allows it to capture complex patterns while maintaining generalization. The only trade-off is slightly lower recall compared to Decision Tree and Naive Bayes, but the superior precision and overall balance make it the most reliable model for wine quality prediction. |

## Project Structure

```
ml-assignment-2/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # This file
│── test_data.csv           # Test dataset (20% split, 1300 samples)
│── model/
│   ├── train_models.py     # Model training script
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k_nearest_neighbors.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl          # StandardScaler for feature scaling
│   ├── metrics.pkl         # Saved evaluation metrics
│   └── feature_names.pkl   # Feature column names
```

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Bharathraj-N/ml-assignment-2.git
cd ml-assignment-2

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models
python model/train_models.py

# Run Streamlit app
streamlit run app.py
```

## Technologies Used

- Python 3.10+
- Scikit-learn 1.5.1
- Streamlit 1.36.0
- Pandas 2.2.2
- NumPy 1.26.4
- Matplotlib 3.9.1
- Seaborn 0.13.2
