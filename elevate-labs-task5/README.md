# Task 5 — Decision Trees and Random Forests

## Objective
Build tree-based classification models to classify software requirements as Functional or Non-Functional, analyze overfitting, and compare a single Decision Tree against a Random Forest.

## Dataset
`software_requirements_cleaned.csv` — the same cleaned dataset used in Tasks 1, 2, 3, and 4.
It contains 527 software requirements with columns: Type, Description, Type_encoded, requirement_length, requirement_length_scaled.

## What I Did

### Binary Target Creation
- **Input (X):** `requirement_length` and `requirement_length_scaled`
- **Output (y):** `is_functional` — 1 if the requirement is Functional (F), 0 for all other types (Non-Functional)
- Split data 80% train / 20% test with stratification

### Overfitting Analysis
- Trained a full unrestricted Decision Tree — Train Accuracy: 0.7411, Test Accuracy: 0.5755
- The large gap confirms overfitting — the tree memorised training data
- Swept max_depth from 1 to 15 and plotted Train vs Test accuracy at each depth
- Best depth auto-detected at **depth=7** (Test Accuracy: 0.6981)

### Decision Tree (Pruned)
**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 0.6981 |
| CV Mean (5-fold) | 0.5635 ± 0.0321 |

### Random Forest
- Trained with 100 estimators (`n_estimators=100`)

**Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | 0.6038 |
| CV Mean (5-fold) | 0.5427 ± 0.0373 |

The pruned Decision Tree outperformed the Random Forest on this dataset.
This is expected — Random Forests need larger datasets to outshine a single tree.
With only 527 rows, the ensemble doesn't have enough variety to gain an advantage.

### Feature Importances
- The Decision Tree relied more on `requirement_length` (~0.6) than `requirement_length_scaled` (~0.4)
- The Random Forest split importance almost equally between both features (~0.5 each)
- Since both features encode the same underlying signal at different scales, this split is expected

### Cross-Validation
- Used 5-fold CV to get a more honest accuracy estimate than a single train/test split
- DT CV mean (0.5635) is lower than single-split test accuracy (0.6981), suggesting some variance in the data
- RF CV mean (0.5427) was consistently slightly below the DT across all folds

## Tools Used
- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Files
```
elevate-labs-task5/
├── data/
│   └── software_requirements_cleaned.csv
├── notebook/
│   └── Task_05.ipynb
├── images/
│   ├── overfitting_analysis.png
│   ├── decision_tree_viz.png
│   ├── dt_confusion_matrix.png
│   ├── rf_confusion_matrix.png
│   ├── feature_importances.png
│   └── cross_validation.png
└── README.md
```

## Key Learnings
- A Decision Tree with no depth limit memorises training data — this is overfitting
- Controlling `max_depth` is the primary way to prune a tree and reduce overfitting
- Random Forests reduce overfitting by averaging many trees trained on random data subsets (bagging)
- Feature importance scores show how much each feature reduced impurity across all splits — values sum to 1
- Cross-validation gives a more reliable accuracy estimate than a single train/test split
- On small datasets, a well-tuned single tree can outperform an ensemble
