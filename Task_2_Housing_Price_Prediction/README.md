# Real Estate Housing Price Prediction

### Machine Learning | Regression | Feature Engineering | Model Optimization

An end-to-end Machine Learning project focused on predicting median house values using the California Housing Dataset. The project explores data preprocessing, feature engineering, multicollinearity analysis, regression algorithms, and hyperparameter optimization to develop and evaluate predictive models.

---

## 1. Project Overview

Accurately estimating property prices is an important application of predictive analytics in the real estate industry. Housing prices are influenced by several factors, including geographical location, household income, population, housing characteristics, and proximity to the ocean.

The objective of this project is to develop regression models that predict median house values based on these features.

Three regression algorithms are implemented and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression

The project also incorporates GridSearchCV for hyperparameter optimization and evaluates model performance using multiple regression metrics.

## 2. Project Objectives

The primary objectives of this project are:

- Understand the dataset through Exploratory Data Analysis (EDA).
- Identify and handle missing values.
- Perform data preprocessing using appropriate imputation and transformation techniques.
- Create interaction features to capture relationships between variables.
- Identify multicollinearity using Variance Inflation Factor (VIF).
- Implement and compare multiple regression algorithms.
- Optimize model hyperparameters using GridSearchCV.
- Evaluate predictive performance using RMSE, MAE, and R².
- Analyze prediction errors through actual-versus-predicted and residual visualizations.

## 3. Dataset Description

**Dataset:** California Housing Dataset

**Source:** [California Housing Dataset – Hands-On Machine Learning](https://github.com/ageron/handson-ml2)

The dataset was downloaded and loaded locally from a CSV file.

**Dataset File:** `California_Housing_Dataset.csv`

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Total Records | 20,640 |
| Total Columns | 10 |
| Numerical Features | 9, including the target |
| Categorical Features | 1 |
| Target Variable | `median_house_value` |
| Missing Values | 207 |

### Feature Description

| Feature | Description |
|---|---|
| longitude | Geographic longitude of the district |
| latitude | Geographic latitude of the district |
| housing_median_age | Median age of houses in the district |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the district |
| households | Number of households |
| median_income | Median income of households |
| ocean_proximity | Proximity of the district to the ocean |
| median_house_value | Median house value (prediction target) |

**Target Variable:** `median_house_value`

## 4. Technologies and Libraries

The project was implemented using Python and the following libraries:

| Technology / Library | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development and experimentation |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model development and evaluation |
| Statsmodels | Statistical analysis and VIF calculation |

## 5. Project Workflow

### 5.1 Exploratory Data Analysis (EDA)

Performed an initial analysis to understand the structure, characteristics, and distribution of the dataset.

Key activities included:

- Inspecting the first and last records.
- Examining dataset dimensions and column information.
- Analyzing descriptive statistics.
- Identifying missing values and duplicate records.
- Examining numerical and categorical features.
- Visualizing the target variable distribution.
- Studying correlations between numerical variables.

### 5.2 Data Preprocessing

Prepared the dataset for machine learning through the following steps:

- Separated independent features and the target variable.
- Divided the dataset into training and testing subsets using an 80:20 split.
- Applied median imputation to handle missing numerical values.
- Applied most-frequent imputation to categorical data.
- Used StandardScaler for numerical feature scaling.
- Applied OneHotEncoder to transform categorical variables into numerical representations.

Preprocessing operations were organized using Scikit-learn Pipelines and ColumnTransformer to maintain a consistent transformation workflow and prevent data leakage during model training.

### 5.3 Feature Engineering

Created interaction features to capture potential relationships between selected numerical variables.

The following features were selected:

- `median_income`
- `housing_median_age`
- `total_rooms`

Used `PolynomialFeatures` with:

- Degree: 2
- Interaction Only: True
- Include Bias: False

This generated pairwise interaction features without adding squared terms.

### 5.4 Multicollinearity Analysis using VIF

Variance Inflation Factor (VIF) was calculated to identify multicollinearity among numerical independent variables.

The analysis highlighted relatively high VIF values for:

- `total_bedrooms`
- `households`
- `total_rooms`

The results were used to understand relationships among the predictors and assess potential multicollinearity.

### 5.5 Regression Model Development

Three regression algorithms were implemented.

**Linear Regression**

Used as the baseline model to establish the initial predictive performance.

**Ridge Regression**

Applied L2 regularization to reduce the influence of large regression coefficients and address potential multicollinearity.

**Lasso Regression**

Applied L1 regularization, which can shrink some feature coefficients to exactly zero and perform implicit feature selection.

### 5.6 Hyperparameter Optimization using GridSearchCV

GridSearchCV was applied to Ridge and Lasso Regression to identify suitable regularization strengths.

**Configuration:**

- Alpha values: `[0.001, 0.01, 0.1, 1, 10, 100]`
- Cross-validation: 5-fold
- Scoring metric: Negative Root Mean Squared Error
- Optimization technique: Grid Search

The best-performing configurations were selected using cross-validation on the training data.

### 5.7 Model Evaluation

The models were evaluated using three standard regression metrics:

| Metric | Purpose | Preferred Value |
|---|---|---|
| RMSE | Measures the square root of average squared prediction errors | Lower |
| MAE | Measures average absolute prediction errors | Lower |
| R² Score | Measures the proportion of target variance explained by the model | Higher |

### 5.8 Model Visualization and Error Analysis

The following visualizations were generated to understand the dataset and evaluate model performance:

- Target variable distribution.
- Correlation heatmap.
- Variance Inflation Factor (VIF) comparison.
- RMSE comparison across regression models.
- MAE comparison across regression models.
- R² score comparison.
- Actual vs. Predicted house prices.
- Residual error plot.
- Residual distribution plot.

These visualizations help assess model performance, prediction accuracy, and the distribution of errors.

## 6. Model Performance

The following table presents the initial test-set evaluation results using the default regularization strength of `alpha = 1.0` for Ridge and Lasso Regression.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Linear Regression | **69,709.75** | **49,897.43** | **0.629165** |
| Ridge Regression | 69,712.37 | 49,901.68 | 0.629138 |
| Lasso Regression | 69,710.16 | 49,898.49 | 0.629161 |

### Initial Model Evaluation

Linear Regression achieved the lowest RMSE and MAE, along with the highest R² score among the three initial models.

The models demonstrated very similar predictive performance, indicating that the initial regularization settings did not produce a substantial improvement over the baseline model.

*Note: The above table represents initial model results. Hyperparameter-tuned Ridge and Lasso models were evaluated separately using GridSearchCV.*

## 7. Key Findings

- The dataset contains 207 missing values in the `total_bedrooms` feature, which were handled through median imputation.
- Feature engineering introduced pairwise interaction features to capture additional relationships.
- VIF analysis identified multicollinearity among several numerical predictors.
- Linear Regression, Ridge Regression, and Lasso Regression produced comparable initial results.
- GridSearchCV was used to optimize the regularization parameters of Ridge and Lasso Regression.
- Actual-versus-predicted and residual plots provided additional insight into prediction errors and model behavior.

## 8. Project Structure

The repository is organized as follows:

```text
Task_2_Housing_Price_Prediction/
│
├── housing_price_prediction.ipynb
│
├── California_Housing_Dataset.csv
│
├── README.md
│
└── requirements.txt
```

## 9. Installation and Execution

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd Task_2_Housing_Price_Prediction/
```

### Step 3: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the Notebook

Open `housing_price_prediction.ipynb` and execute the cells sequentially.

**Important:** Keep `California_Housing_Dataset.csv` in the same directory as the notebook, or update the dataset path in the notebook if the file is stored elsewhere.

## 10. Requirements

The project dependencies are listed in `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
jupyter
```

## 11. Skills Demonstrated

This project demonstrates practical knowledge and application of:

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Statistical Analysis
- Multicollinearity Analysis
- Regression Modeling
- Regularization Techniques
- Hyperparameter Optimization
- Cross-Validation
- Model Evaluation
- Data Visualization
- Machine Learning Pipelines

## 12. Conclusion

This project demonstrates an end-to-end regression-based machine learning workflow for predicting median house values using the California Housing Dataset.

Starting with exploratory data analysis and data preprocessing, the implementation progresses through feature engineering, multicollinearity analysis, regression model development, and hyperparameter optimization.

Linear Regression, Ridge Regression, and Lasso Regression were evaluated using RMSE, MAE, and R² score. GridSearchCV was further applied to optimize the regularization parameters of Ridge and Lasso Regression.

The initial evaluation showed that Linear Regression achieved slightly better performance than the other baseline models, with an RMSE of 69,709.75 and an R² score of approximately 0.6292.

Overall, the project provided practical experience in building, evaluating, comparing, and interpreting machine learning regression models.

## 13. Acknowledgement

This project was completed as part of the **Datheon Machine Learning & Data Science Virtual Engineering Residency – Task 2: Real Estate Housing Price Prediction**.

Machine Learning & Data Science Enthusiast

---

*If you find this project interesting, feel free to explore the notebook and its implementation.*