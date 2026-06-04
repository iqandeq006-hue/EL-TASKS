# Task 4 — Classification with Logistic Regression

## Objective
Build a binary classifier using Logistic Regression to classify software requirements as Functional or Non-Functional.

## Dataset
`software_requirements_cleaned.csv` — the same cleaned dataset used in Tasks 1, 2, and 3.
It contains 527 software requirements with columns: Type, Description, Type_encoded, requirement_length, requirement_length_scaled.

## What I Did

### Binary Target Creation
- **Input (X):** `requirement_length` and `requirement_length_scaled`
- **Output (y):** `is_functional` — 1 if the requirement is Functional (F), 0 for all other types (Non-Functional)
- Split data 80% train / 20% test with stratification
- Standardized features using StandardScaler
- Fitted a LogisticRegression model from scikit-learn

**Results:**
| Metric | Value |
|--------|-------|
| ROC-AUC Score | 0.7083 |
| True Negatives (TN) | 64 |
| False Positives (FP) | 2 |
| False Negatives (FN) | 35 |
| True Positives (TP) | 5 |

The model correctly identifies Non-Functional requirements well but misses many Functional ones.
This is due to class imbalance — Non-Functional requirements outnumber Functional ones in the dataset.

### Threshold Tuning
- Tested thresholds: 0.3, 0.4, 0.5, 0.6, 0.7
- Lower thresholds improve recall for Functional requirements but reduce precision
- Default threshold of 0.5 was used for final evaluation

### Sigmoid Function
- Visualized σ(z) = 1 / (1 + e^−z)
- Shows how logistic regression maps any real number to a probability between 0 and 1
- Default decision boundary at 0.5 marked on the plot

## Tools Used
- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Files
```
elevate-labs-task4/
├── data/
│   └── software_requirements_cleaned.csv
├── notebook/
│   └── Task_04.ipynb
├── images/
│   ├── sigmoid_curve.png
│   ├── confusion_matrix.png
│   └── roc_auc_curve.png
└── README.md
```

## Key Learnings
- Logistic regression uses the sigmoid function to output probabilities instead of raw values
- Confusion matrix breaks down TP, TN, FP, FN — more informative than accuracy alone
- ROC-AUC of 0.7083 means the model is moderately better than random guessing
- Class imbalance causes the model to favour the majority class (Non-Functional)
- Threshold tuning lets you trade off precision vs recall depending on the use case
