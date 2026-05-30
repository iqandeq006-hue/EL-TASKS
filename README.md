# Task 01 — Data Cleaning & Preprocessing
### Elevate Labs AI/ML Internship | Dharshin R ([@iqandeq006-hue](https://github.com/iqandeq006-hue))

---

## Overview

This task covers the fundamentals of data cleaning and preprocessing — the first and most critical stage of any machine learning pipeline.

Rather than using the generic Titanic dataset, I chose the **PROMISE NFR (Non-Functional Requirements) Dataset** — a real-world software requirements dataset. This choice is intentional: it directly connects to my ongoing project work on an **LLM-based Voice-to-SRS Document Generator**, where an agent needs to classify and structure software requirements automatically.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | Software Requirements Dataset (PROMISE NFR) |
| **Source** | [Kaggle — iamsouvik/software-requirements-dataset](https://www.kaggle.com/datasets/iamsouvik/software-requirements-dataset) |
| **Format** | Plain text file (`nfr.txt`) parsed into structured DataFrame |
| **Size** | 625 raw entries → 527 after outlier removal |
| **Columns** | `Type`, `Description` |

### Requirement Types in Dataset

| Code | Meaning |
|---|---|
| `F` | Functional |
| `US` | Usability |
| `SE` | Security |
| `PE` | Performance |
| `O` | Operational |
| `LF` | Look & Feel |
| `SC` | Scalability |
| `A` | Availability |
| `MN` | Maintainability |
| `L` | Legal |
| `FT` | Fault Tolerance |
| `PO` | Portability |

---

## Steps Performed

### 1. Library Setup & Import
Imported `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn` preprocessing tools.

### 2. Dataset Download & Loading
Downloaded via Kaggle API. The dataset was a `.txt` file (not CSV), so a custom line-by-line parser was written to split each entry on `:` and extract `Type` and `Description` fields into a DataFrame.

```python
with open('nfr.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            req_type = parts[0].strip()
            req_description = parts[1].strip()
            requirements.append({'Type': req_type, 'Description': req_description})
```

### 3. Exploratory Data Analysis
- Checked shape, column types, and missing values
- Dataset had no missing values after parsing

### 4. Handling Missing Values
- Text columns: filled with `'Unknown'`
- Numerical columns: filled with median value

### 5. Categorical Encoding
- Used `LabelEncoder` to convert `Type` (12 categories) into numerical `Type_encoded`
- Example: `A→0`, `F→1`, `FT→2`, `L→3`, `LF→4` ...

### 6. Feature Engineering + Normalization
- Created `requirement_length` column — character count of each requirement description
- Applied `StandardScaler` to produce `requirement_length_scaled` (mean=0, std=1)

### 7. Outlier Detection & Removal
- Visualized distribution using boxplot
- Applied IQR method (1.5× fence) to remove outliers
- Reduced from **625 → 527 rows**

---

## Output

Cleaned dataset saved as `software_requirements_cleaned.csv` with 5 columns:

```
Type | Description | Type_encoded | requirement_length | requirement_length_scaled
```

---

## Folder Structure

```
elevate-labs-task1/
│
├── data/
│   ├── nfr.txt                             # original raw dataset
│   └── software_requirements_cleaned.csv   # cleaned output
│
├── notebook/
│   └── Task_01.ipynb                       # full preprocessing notebook
│
├── images/
│   └── boxplot_outliers.png                # outlier visualization
│
└── README.md
```

---

## Why This Dataset?

My current project goal is building a **Voice-to-SRS Document Generator** — an LLM agent that:
1. Listens to a user describe a software project via voice
2. Identifies what information is still needed
3. Asks targeted questions to fill an SRS document structure
4. Classifies each requirement as functional or non-functional
5. Outputs a completed SRS document

This dataset is exactly what that agent will work with. Preprocessing it here means I understand the data my agent will eventually classify and structure automatically.

---

## Tools Used

- Python 3
- Google Colab
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LabelEncoder, StandardScaler)
- Kaggle API

---

## Internship

**Program:** Elevate Labs — 45 Day AI/ML Internship  
**Task:** 01 — Data Cleaning & Preprocessing  
**Submitted by:** Dharshin R | [@iqandeq006-hue](https://github.com/iqandeq006-hue)
