# Breast Cancer Classification

A machine learning project that classifies breast tumors as **malignant** or **benign** using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer` (built into scikit-learn, no download needed)
- **Samples:** 569
- **Features:** 30 numeric features describing cell nucleus properties (radius, texture, smoothness, etc.)
- **Classes:** Malignant (212 samples) / Benign (357 samples)

## Workflow

1. Load and explore the data
2. Exploratory Data Analysis (class distribution, feature distributions, correlation heatmap)
3. Preprocessing (80/20 train-test split with stratification, feature scaling with `StandardScaler`, fit only on training data to avoid leakage)
4. Train and compare 4 models: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest
5. Evaluate the best model with a confusion matrix and classification report (precision, recall, F1)

## Results

All four models perform well on this dataset. The best model is selected by test accuracy, with special attention to **recall on the malignant class**, since missing an actual cancer case (false negative) is the most costly error in a medical context.

See the notebook for the full comparison table, confusion matrix, and classification report.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_classification.ipynb
```

## Project structure

```
.
├── breast_cancer_classification.ipynb   # Main notebook
├── requirements.txt                     # Python dependencies
└── README.md
```

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn
