# Rossmann Store Sales Forecasting with XGBoost

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EB0028)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-success)
![License](https://img.shields.io/badge/license-MIT-blue)

## Project Description

A machine learning project that forecasts daily sales for **1,115 Rossmann drugstores** across 7 European countries using gradient boosting. Built on the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales) dataset (1M+ observations).

**Why it exists.** Store managers need forecast sales individually for the next 6 weeks. However, because of variation features in each store, results are inconsistent across the chain. The demand of a centralized model integrates promotions, competition, holidays, and seasonality to ensure more accurate forecast, efficient inventory and decision-making. 

**Key objectives**
- Engineer features from raw sales, store, and promotion data.
- Train an XGBoost regressor to predict daily sales up to 6 weeks ahead.
- Validate model stability with K-Fold cross-validation.
- Identify the strongest sales drivers via feature importance.

## What I Did

- **Merged and cleaned** two datasets (1.01M sales records × 1,115 store profiles) into a single training frame.
- **Filtered noise** by removing 172,817 closed-store records where sales were zero. 
- **Built a preprocessing pipeline** handling missing values, scaling numeric features, and one-hot encoding categorical variables.
- **Trained an XGBoost regressor** and validated it with 10-fold cross-validation for a robust performance estimate.
- **Analyzed feature importance** to identify the top drivers of sales — actionable insight for the business.
- **Generated a Kaggle-ready submission** file, applying the closed-store rule to zero out predictions where stores are shut.

## Features

- End-to-end pipeline: data cleaning → feature engineering → modeling → evaluation.
- Custom feature engineering: date decomposition, competition duration, Promo2 activity flags.
- Preprocessing pipeline with `MinMaxScaler` (numerical) and `OneHotEncoder` (categorical).
- 10-fold cross-validation for robust RMSPE estimation.
- Feature importance analysis to surface the strongest sales drivers.
- Kaggle-ready submission file generation.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy |
| ML | XGBoost, scikit-learn |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook, Anaconda |

## Project Structure

```
rossmann-sales-forecasting/
├── README.md
├── ApproachML.ipynb
├── rossmann-store-sales/
│   ├── train.csv
│   ├── store.csv
│   └── test.csv
└── submission.csv
```

## Install and Run

```bash
git clone https://github.com/QuocPhan555/Sales-store_Gradient-Boosting.git
cd Sales-store_Gradient-Boosting
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost opendatasets jupyter
```

Download the dataset from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) or get zip file from my main, place `train.csv`, `store.csv`, and `test.csv` in `rossmann-store-sales/`, then:

```bash
jupyter notebook ApproachML.ipynb
```

## How to Use

1. Open `ApproachML.ipynb` and run cells sequentially.
2. The notebook merges `train.csv` with `store.csv`, engineers date and promotion features, then fits an XGBoost regressor.
3. Predictions on the test set are written to `submission.csv` — ready for direct Kaggle submission.
4. Adjust `max_depth`, `n_estimators`, or the feature list in the modeling section to experiment.

**Selected results**
- Model: `XGBRegressor(max_depth=4, n_estimators=20)`
- 10-fold CV validation RMSE: ~2,200–2,660 (mean ≈ 2,430)
- Top sales drivers: **Promo** (33% importance), day-of-week, store type, and Promo2 participation.

---
*Phan Thanh Anh Quoc — M2 Advanced Applied Economics, University of Caen Normandy*
