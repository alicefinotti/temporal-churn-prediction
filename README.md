# Temporal Churn Prediction

## Overview

Predict user churn within a **10-day horizon** using event-level activity from a music streaming service. The analysis processes approximately **17.5 million events**, constructs rolling user-level snapshots, and addresses a heavily imbalanced outcome (approximately 5.2% positive observations).

## Methodology

- **Behavioral feature engineering:** Aggregate listening activity, sessions, advertising exposure, skips, negative feedback, recency, and activity trends, including 3-, 7-, and 14-day windows.
- **Temporal snapshots:** Build snapshots at seven-day intervals, using only events available by each cutoff date for features and subsequent events for churn labels.
- **Modeling:** Train two XGBoost classifiers for 10-day churn and eventual churn, with class weighting and Optuna hyperparameter tuning.
- **Validation:** Hold out the most recent cutoff dates for evaluation, use GroupKFold to separate users during tuning, and fit preprocessing parameters within each training fold.
- **Probability blending:** Combine the classifiers' predicted probabilities, selecting the blend weight through out-of-fold balanced accuracy on the 10-day churn target.

## Results

The accompanying report documents the study's model comparisons and historical results, including an ROC AUC of **0.8072**, balanced accuracy of **0.7195**, and F1-score of **0.2330**.
