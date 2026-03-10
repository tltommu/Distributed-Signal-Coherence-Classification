
# Distributed Signal Coherence Classification – Project Pipeline

## Overview

This project implements a machine learning pipeline for the **Distributed Signal Coherence Classification** competition.

The objective is to predict whether a signal sample contains **signal incoherence** using engineered numerical features.

The pipeline focuses on:

- Robust cross-validation
- Handling extreme class imbalance
- Matthews Correlation Coefficient (MCC) optimization
- Competition-oriented prediction strategies

The final model uses **LightGBM with stratified cross-validation**, threshold optimization, and **Top-K prediction selection**.

---

# Dataset

The dataset contains two main files:

## Training Dataset
- Multiple signal-related numerical features
- Target column: `signal_incoherence_flag`

## Test Dataset
- Same feature structure as training data
- Used to generate prediction submissions

### Key Characteristics

- **Highly imbalanced classification problem**
- Positive class ≈ **3% of samples**
- Includes several **random noise features**

Because of the imbalance, the model uses **class weighting and MCC-based optimization**.

---

# Project Pipeline

The full modeling workflow:

```

Load Dataset
↓
Feature Selection
↓
Stratified Cross Validation
↓
LightGBM Training
↓
Out-of-Fold Predictions
↓
MCC Threshold Optimization
↓
Top-K Prediction Strategy
↓
Project Submission

````

---

# Model

The primary model used is **LightGBM**.

LightGBM was selected because it:

- Performs strongly on tabular datasets
- Handles nonlinear relationships
- Captures feature interactions
- Provides fast training with early stopping

## Model Parameters

```python
LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    scale_pos_weight=20,
    random_state=42
)
````

### Techniques Used

* Stratified K-Fold cross validation
* Class imbalance weighting
* Early stopping
* Out-of-fold prediction evaluation

---

# Evaluation Metric

The competition uses **Matthews Correlation Coefficient (MCC)**.

MCC is particularly suitable for **imbalanced datasets** because it considers:

* True positives
* True negatives
* False positives
* False negatives

The Kaggle leaderboard uses a **scaled MCC**:

```
Scaled Score = (MCC + 1) / 2
```

---

# Threshold Optimization

Instead of using a default probability threshold (0.5), the pipeline searches for the threshold that maximizes MCC.

Threshold search range:

```
0.01 → 0.20
```

Example result:

```
Best MCC: 0.13970
Scaled Score: 0.56985
Optimal Threshold: 0.07
```

---

# Top-K Prediction Strategy

Because the dataset contains **only ~3% positive samples**, predictions are converted to binary labels using a **Top-K strategy**.

### Steps

1. Estimate the positive rate from training data
2. Rank prediction probabilities
3. Select the top K samples as positive

```
K = len(test) × positive_rate
```

This method often performs better than a fixed threshold in ranking-based competitions.

---

# Cross Validation Setup

```
Stratified K-Fold
Number of folds: 7
Random seed: 42
```

Stratification ensures that each fold maintains the original class distribution.

---

# Model Diagnostics

Example training statistics:

```
OOF min: 0.019
OOF max: 0.217
OOF mean: 0.036
Positive rate: 0.029

Best MCC: 0.13970
Scaled Score: 0.56985
Optimal Threshold: 0.07
```

These statistics indicate that the model predictions are **well calibrated** relative to the dataset imbalance.

---

# Results

| Metric               | Score       |
| -------------------- | ----------- |
| Cross Validation MCC | **0.1397**  |
| Scaled MCC           | **0.5699**  |
| Project Public Score  | **~0.5168** |

Differences between cross-validation and leaderboard scores can occur due to:

* Test distribution differences
* Dataset variance
* Cross-validation randomness

---

# Techniques Used

* LightGBM gradient boosting
* Stratified cross-validation
* Class imbalance weighting
* MCC threshold optimization
* Top-K prediction strategy
* Out-of-fold prediction evaluation

---

# Future Improvements

Potential improvements for the pipeline include:

* Feature importance based filtering
* Permutation importance noise removal
* Model stacking (LightGBM + Logistic Regression)
* Fold-stable feature selection
* Hyperparameter tuning
* Advanced feature engineering

---

# Requirements

The project uses the following Python libraries:

```
pandas
numpy
scikit-learn
lightgbm
```

Install dependencies with:

```
pip install pandas numpy scikit-learn lightgbm
```

---

# Running the Pipeline

Train the model pipeline:

```
python train_pipeline.py
```

Generate the submission file:

```
submission.csv
```


# Final Spoiler

⚠️ Competition Twist

All the feature columns were intentionally anonymized during the competition to ensure participants focused on true data science techniques rather than domain knowledge. After the competition ended, the actual meaning of the dataset was revealed.

The dataset represented a house auction dataset, where the features corresponded to attributes such as:

- House location
- Property size
- Property price
- Other housing characteristics
  
Participants were therefore solving what was essentially a real estate prediction problem, but with the feature meanings hidden to simulate a pure machine learning challenge.

---

# Author

This repository contains a **competition-style machine learning pipeline** designed for signal classification under heavy class imbalance.

The project emphasizes **robust evaluation, careful threshold selection, and practical modeling techniques**.
