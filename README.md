# Insurance Premiums Regression

This project aims to predict insurance premiums using a variety of machine learning models, including Decision Trees, Random Forests, Neural Networks, XGBoost, and CatBoost. The dataset is sourced from the Kaggle competition.

## Project Overview

The goal of this project is to build a regression model that predicts insurance premiums based on a mix of numerical and categorical features. The project explores various machine learning techniques, including hyperparameter tuning and feature engineering, to optimize model performance.

## Dataset

The dataset consists of two files:
- `train.csv`: Contains 1,200,000 rows and 21 columns, including the target variable `target`.
- `test.csv`: Contains 800,000 rows and 20 columns, excluding the target variable.

### Key Features:
- **Numerical Features**: `Age`, `feature_0`, `feature_5`, `feature_8`, `feature_10`, etc.
- **Categorical Features**: `Gender`, `feature_1`, `feature_3`, `feature_4`, etc.

## Preprocessing

### Steps:
1. **Handle Missing Values**:
   - Numerical features were filled using random sampling between Q1 and Q3 or the median.
   - Categorical features were filled using random sampling based on value distributions.

2. **Remove Irrelevant Features**:
   - Dropped `id` and `feature_12` as they do not contribute to the target.

3. **Encoding**:
   - Ordinal features were label-encoded.
   - Nominal features were one-hot encoded.

4. **Scaling**:
   - Applied Min-Max Scaling, Standardization, and Robust Scaling based on feature characteristics.

5. **Handle Duplicates**:
   - Removed duplicate rows from the training dataset.

## Feature Engineering

### Techniques:
- **Mathematical Transformations**: Generated sine, cosine, tangent, and other transformations for numerical features.
- **Arithmetic Combinations**: Created new features by combining top numerical features.
- **Clustering**: Added cluster labels using KMeans.
- **Feature Selection**: Identified important features using model-specific importance metrics.

## Models

### 1. **Decision Tree**
- Initial and tuned models were evaluated using Bayesian optimization.
- Hyperparameters tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`, etc.

### 2. **Random Forest**
- Ensemble model with hyperparameter tuning.
- Hyperparameters tuned: `n_estimators`, `max_depth`, `max_features`, etc.

### 3. **Neural Network**
- Implemented using PyTorch and integrated with `skorch`.
- Hyperparameters tuned: `hidden_layers`, `learning_rate`, `batch_size`, etc.

### 4. **XGBoost**
- Gradient boosting model with GPU acceleration.
- Hyperparameters tuned: `max_depth`, `learning_rate`, `n_estimators`, etc.

### 5. **CatBoost**
- Gradient boosting model optimized for categorical features.
- Hyperparameters tuned: `iterations`, `learning_rate`, `depth`, etc.


## Performance

### Metrics:
- **Root Mean Squared Logarithmic Error (RMSLE)** was used as the primary evaluation metric.

### Summary:
| Model          | Train RMSLE | Valid RMSLE | Test RMSLE |
|----------------|-------------|-------------|------------|
| Decision Tree  | 1.1596      | 1.1596      | 1.1626     |
| Random Forest  | 1.1545      | 1.1578      | 1.1611     |
| Neural Network | 1.1653      | 1.1653      | 1.1656     |
| XGBoost        | 1.1256      | 1.1256      | -          |
| CatBoost       | 2.0077      | 2.0086      | -          |

---

## How to Run

### Prerequisites:
- Python 3.8 or higher
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `catboost`, `torch`, `skorch`, `scipy`, `scikit-optimize`

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/insurance-premiums-regression.git
   cd insurance-premiums-regression
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook: Open `main.ipynb` in Jupyter Notebook or Visual Studio Code and execute the cells sequentially.
