# House Price Prediction using Linear Regression

A machine learning project that predicts house prices based on property features such as area, bedrooms, bathrooms, stories, parking, and amenities. Built as part of my internship task at **SoftGrowthTech**.

## 📌 Project Overview

This project implements a complete machine learning pipeline:
- Load and explore the dataset
- Clean and preprocess the data (encode categorical features)
- Train a Linear Regression model
- Predict house prices from new input features
- Evaluate model performance using standard regression metrics

## 📊 Dataset

**Source:** [Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

The dataset contains 545 house records with the following features:

| Feature | Description |
|---|---|
| area | Square footage of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Whether the house is connected to the main road (Yes/No) |
| guestroom | Whether the house has a guest room (Yes/No) |
| basement | Whether the house has a basement (Yes/No) |
| hotwaterheating | Whether the house has hot water heating (Yes/No) |
| airconditioning | Whether the house has air conditioning (Yes/No) |
| parking | Number of parking spaces |
| prefarea | Whether the house is in a preferred area (Yes/No) |
| furnishingstatus | Furnished, Semi-furnished, or Unfurnished |
| **price** | Target variable — sale price of the house |

## 🛠️ Tech Stack

- **Python**
- **Pandas** — data loading & manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — model training & evaluation
- **Matplotlib** — visualization

## 🔄 Workflow

### 1. Data Loading
Loaded the dataset using `pandas.read_csv()` and inspected shape, data types, and missing values. Confirmed 545 rows, 13 columns, no missing values.

### 2. Data Preprocessing & Cleaning
- Encoded 6 Yes/No columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) to binary 1/0
- One-hot encoded `furnishingstatus` (3 categories, no natural order), using `drop_first=True` to avoid the dummy variable trap
- Confirmed no missing values and all columns numeric before modeling

### 3. Model Training
- Split data into 80% training / 20% testing sets (`train_test_split`, `random_state=42`)
- Trained a **Linear Regression** model using scikit-learn
- Inspected feature coefficients to validate the model learned sensible relationships (e.g. `area`, `airconditioning`, and `prefarea` all positively associated with price)

### 4. Prediction
Built a `predict_house_price()` function that takes new input features and returns a predicted price.

### 5. Model Evaluation

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | ₨970,043.40 |
| MSE (Mean Squared Error) | ₨1,754,318,687,330.66 |
| RMSE (Root Mean Squared Error) | ₨1,324,506.96 |
| **R² Score** | **0.6529** |
| **Average Prediction Accuracy** | **80.63%** |

## 📈 Results & Insights

The model achieved an **R² score of 0.6529**, meaning it explains roughly 65% of the variance in house prices using the available features — a solid result for a Linear Regression baseline on real housing data. The average prediction accuracy (based on MAE relative to mean price) was **80.63%**.

Examining the feature coefficients confirmed the model learned realistic, real-world relationships: larger area, more bathrooms, air conditioning, and a preferred location all increased predicted price, while an unfurnished status decreased it — consistent with how the real housing market behaves.

The remaining ~35% of unexplained variance is expected, since house prices in reality also depend on factors not captured in this dataset (exact location/neighborhood, market timing, renovation quality, etc.). This is a known and accepted limitation of Linear Regression on a relatively small (545-row) dataset.

**Possible next steps:** try regularized regression (Ridge/Lasso) to handle multicollinearity between features, or test ensemble models (Random Forest, Gradient Boosting) to capture non-linear relationships.

## 📁 Repository Structure

```
house-price-prediction/
├── house_price_prediction.ipynb     # Main notebook (full pipeline)
├── housing.csv                       # Dataset
├── actual_vs_predicted.png           # Evaluation visualization
└── README.md
```

## 🚀 How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
3. Open `house_price_prediction.ipynb` in Jupyter Notebook
4. Run all cells in order

## 🎓 Internship

This project was completed as part of my Machine Learning internship at **SoftGrowthTech**.

---
**Author:** Muhammad Kaif
**GitHub:** [MuhammadKaif-112](https://github.com/MuhammadKaif-112)
