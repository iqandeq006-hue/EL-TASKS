# Task 3 — Linear Regression

## Objective
Implement and understand simple and multiple linear regression using a software requirements dataset.

## Dataset
`software_requirements_cleaned.csv` — the same cleaned dataset used in Tasks 1 and 2.
It contains 527 software requirements with columns: Type, Description, Type_encoded, requirement_length, requirement_length_scaled.

## What I Did

### Simple Linear Regression
- **Input (X):** `Type_encoded` — the requirement type as a number (0–11)
- **Output (y):** `requirement_length` — how many characters the requirement description has
- Split data 80% train / 20% test
- Fitted a LinearRegression model from scikit-learn

**Results:**
| Metric | Value |
|--------|-------|
| MAE | 30.41 |
| MSE | 1774.39 |
| R² | 0.1011 |

The model explains ~10% of variance — Type alone is a weak predictor of length.

### Multiple Linear Regression
- **Inputs (X):** `Type_encoded` + `word_count` (calculated from Description, not saved to CSV)
- **Output (y):** `requirement_length`

**Results:**
| Metric | Value |
|--------|-------|
| MAE | 8.10 |
| MSE | 117.83 |
| R² | 0.9403 |

Adding word_count jumps R² from 0.10 to 0.94 — word count is a very strong predictor of character length.

**Coefficients:**
- Type_encoded: 0.3045 (small effect)
- word_count: 5.7016 (each extra word adds ~5.7 characters on average)

## Tools Used
- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Files
```
elevate-labs-task3/
├── data/
│   └── software_requirements_cleaned.csv
├── notebook/
│   └── Task_03.ipynb
├── images/
│   ├── regression_line.png
│   └── residual_plot.png
└── README.md
```

## Key Learnings
- Simple regression with one weak feature gives low R²
- Multiple regression with a stronger second feature (word_count) dramatically improves the model
- MAE tells you average error in real units (characters); R² tells you how well the model explains the data
- word_count does not need to be saved to the CSV — it can be calculated in memory during modelling
