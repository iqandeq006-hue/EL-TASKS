# Task 6 — K-Nearest Neighbors (KNN) Classification

## Objective
Understand and implement the KNN algorithm for classification, experiment with different values of K, evaluate the model, and visualize decision boundaries.

## Dataset
`software_requirements_cleaned.csv` — the same cleaned dataset used in Tasks 1–5.
It contains 527 software requirements with columns: Type, Description, Type_encoded, requirement_length, requirement_length_scaled.

## What I Did

### Binary Target Creation
- **Input (X):** `requirement_length` and `requirement_length_scaled`
- **Output (y):** `is_functional` — 1 if the requirement is Functional (F), 0 for all other types (Non-Functional)
- Features normalized using `StandardScaler` — essential for KNN since it relies on distance calculations
- Split data 80% train / 20% test with stratification

### K Selection — Finding the Best K
- Swept K from 1 to 30 and recorded both Train and Test accuracy at each step
- Plotted Train vs Test accuracy to identify the elbow point
- Best K auto-detected based on highest test accuracy
- K=1 overfits (memorises training data); larger K values underfit (too smoothed)

### Best KNN Model Results
| Metric | Value |
|--------|-------|
| Best K | 15 |
| Train Accuracy | 0.6604 |
| Test Accuracy | 0.6981 |
| CV Mean (5-fold) | 0.5635 ± 0.0660 |

### Confusion Matrix
- Generated for the best K to show True Positives, True Negatives, False Positives, and False Negatives
- Helps understand where the model confuses Functional vs Non-Functional requirements

### Decision Boundary Visualization
- Plotted the decision boundary of the best KNN model on the 2D feature space
- Compared boundaries at K=1 (overfit), best K, and K=25 (underfit) side by side
- K=1 produces jagged, noisy boundaries; higher K produces smoother, more generalized regions

### Distance Metric Comparison
- Evaluated Euclidean, Manhattan, and Chebyshev distance metrics at the best K
- All three metrics were compared on test accuracy to identify which distance measure suits this dataset best

| Distance Metric | Test Accuracy |
|-----------------|---------------|
| Euclidean | 0.6981 |
| Manhattan | 0.6981 |
| Chebyshev | 0.6887 |

### Cross-Validation
- Used 5-fold CV to get a more reliable accuracy estimate than a single train/test split
- Plotted CV scores per fold with a mean line overlay

## Tools Used
- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Files
```
elevate-labs-task6/
├── data/
│   └── software_requirements_cleaned.csv
├── notebook/
│   └── Task_06.ipynb
├── images/
│   ├── k_vs_accuracy.png
│   ├── confusion_matrix.png
│   ├── decision_boundary_best_k.png
│   ├── decision_boundary_comparison.png
│   ├── cross_validation.png
│   └── distance_metrics.png
└── README.md
```

## Key Learnings
- KNN is a lazy learner — it stores all training data and classifies at prediction time by finding the K nearest neighbors
- Normalization is critical for KNN: without it, features with larger ranges dominate the distance calculation
- Choosing K involves a bias-variance tradeoff — small K overfits, large K underfits; the elbow in the test accuracy curve guides the choice
- The time complexity of KNN prediction is O(n × d) per query — it gets slow on large datasets since there is no training phase
- Decision boundaries from KNN are non-linear and can capture complex patterns, unlike linear models
- Cross-validation gives a more honest accuracy estimate than a single train/test split, especially on small datasets
- Distance metric choice (Euclidean vs Manhattan vs Chebyshev) can affect accuracy depending on the data distribution
