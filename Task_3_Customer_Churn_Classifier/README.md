# Customer Churn Prediction

### Machine Learning | Classification | XGBoost | Random Forest | Model Evaluation

An end-to-end Machine Learning project focused on predicting customer churn using the Telco Customer Churn Dataset. The project explores data preprocessing, exploratory data analysis, handling class imbalance, classification algorithms, cross-validation, feature importance, and model performance comparison.

---

## 1. Project Overview

Customer churn refers to customers discontinuing their services or switching to another provider. Predicting customer churn is an important application of machine learning in business analytics, as it helps organizations identify customers who are likely to leave and develop effective retention strategies.

The objective of this project is to develop classification models that predict whether a telecom customer is likely to churn based on customer demographics, subscribed services, account information, and billing details.

Two classification algorithms are implemented and compared:

- Random Forest Classifier
- XGBoost Classifier

The project also incorporates class imbalance handling, Stratified Cross-Validation, and feature importance analysis to evaluate model performance and identify important churn-related factors.

---

## 2. Project Objectives

The primary objectives of this project are:

- Understand the dataset through Exploratory Data Analysis (EDA).
- Analyze customer churn distribution and identify class imbalance.
- Clean and preprocess the dataset.
- Handle missing values using appropriate imputation techniques.
- Transform categorical features using OneHotEncoder.
- Develop classification models using Random Forest and XGBoost.
- Handle class imbalance using class weights and scale_pos_weight.
- Evaluate model performance using multiple classification metrics.
- Perform Stratified 5-Fold Cross-Validation.
- Analyze confusion matrices and ROC curves.
- Identify important features influencing churn predictions.
- Compare the performance of both classification models.

---

## 3. Dataset Description

**Dataset:** Telco Customer Churn Dataset

**Source:** IBM Telco Customer Churn Dataset

**Dataset File:** `Telco_Customer_Churn.csv`

The dataset contains information about telecom customers, including their demographic details, subscribed services, contract types, payment methods, monthly charges, and churn status.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Total Records | 7,043 |
| Total Columns | 21 |
| Independent Features | 20 |
| Target Variable | `Churn` |
| Target Classes | Yes, No |
| Churned Customers | 1,869 |
| Non-Churned Customers | 5,174 |

### Feature Description

| Feature | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Customer gender |
| SeniorCitizen | Whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether phone service is enabled |
| MultipleLines | Whether multiple phone lines are enabled |
| InternetService | Type of internet service |
| OnlineSecurity | Whether online security service is enabled |
| OnlineBackup | Whether online backup service is enabled |
| DeviceProtection | Whether device protection service is enabled |
| TechSupport | Whether technical support is enabled |
| StreamingTV | Whether streaming TV is enabled |
| StreamingMovies | Whether streaming movies is enabled |
| Contract | Type of customer contract |
| PaperlessBilling | Whether paperless billing is enabled |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Monthly charges |
| TotalCharges | Total charges accumulated |
| Churn | Whether the customer discontinued the service |

**Target Variable:** `Churn`

- `No` → 0
- `Yes` → 1

---

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
| Scikit-learn | Preprocessing, classification, and evaluation |
| XGBoost | Gradient boosting classification |

---

## 5. Project Workflow

### 5.1 Exploratory Data Analysis (EDA)

Performed an initial analysis to understand the structure, characteristics, and distribution of the dataset.

Key activities included:

- Inspecting the first records.
- Examining dataset dimensions and column information.
- Analyzing descriptive statistics.
- Identifying missing values and duplicate records.
- Examining numerical and categorical features.
- Analyzing the distribution of the target variable.
- Visualizing customer churn distribution.

### 5.2 Data Preprocessing

Prepared the dataset for machine learning through the following steps:

- Removed the irrelevant customer identifier column.
- Converted `TotalCharges` into a numerical data type.
- Handled missing values using appropriate imputation techniques.
- Separated independent features and target variable.
- Divided the dataset into training and testing subsets using an 80:20 split.
- Used stratified sampling to maintain the target class distribution.
- Applied median imputation to numerical features.
- Applied most-frequent imputation to categorical features.
- Used OneHotEncoder to transform categorical variables into numerical representations.

Preprocessing operations were organized using Scikit-learn Pipelines and ColumnTransformer to maintain a consistent transformation workflow.

### 5.3 Handling Class Imbalance

The dataset contains an unequal distribution of churn and non-churn customers.

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

To address class imbalance, the following techniques were implemented:

- `class_weight="balanced"` in Random Forest.
- `scale_pos_weight` in XGBoost.

These techniques assign greater importance to the minority class during model training.

### 5.4 Classification Model Development

Two classification algorithms were implemented.

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to improve predictive performance and reduce overfitting.

The model was implemented using:

- 100 estimators
- Balanced class weights
- Random state: 42
- Parallel processing

**XGBoost Classifier**

XGBoost is a gradient boosting algorithm that builds decision trees sequentially, with each new tree improving upon the errors of previous trees.

The model was implemented using:

- 100 estimators
- Learning rate: 0.1
- Maximum depth: 3
- Class imbalance adjustment using `scale_pos_weight`
- Evaluation metric: Log Loss
- Random state: 42

### 5.5 Model Evaluation

Both models were evaluated using the following classification metrics:

| Metric | Purpose | Preferred Value |
|---|---|---|
| Accuracy | Overall percentage of correct predictions | Higher |
| Precision | Proportion of predicted churners who actually churned | Higher |
| Recall | Proportion of actual churners correctly identified | Higher |
| F1-Score | Harmonic mean of Precision and Recall | Higher |
| ROC-AUC | Ability to distinguish between the two classes | Higher |

### 5.6 Confusion Matrix Analysis

Confusion matrices were generated for both models to analyze:

- True Positives
- True Negatives
- False Positives
- False Negatives

This analysis helps understand how effectively the models identify churned and non-churned customers.

### 5.7 ROC Curve Analysis

ROC curves were plotted to compare the classification performance of Random Forest and XGBoost across different classification thresholds.

The Area Under the Curve (AUC) score was used to evaluate the ability of each model to distinguish between churn and non-churn customers.

### 5.8 Stratified Cross-Validation

Stratified 5-Fold Cross-Validation was performed on the training dataset.

The dataset was divided into five folds while maintaining the proportion of churn and non-churn customers.

ROC-AUC was used as the scoring metric to evaluate model performance across multiple validation folds.

Cross-validation provides a more reliable estimate of model performance and helps assess model stability.

### 5.9 Feature Importance Analysis

Feature importance scores were extracted from the trained Random Forest and XGBoost models.

The top 10 most important features were visualized using horizontal bar charts.

This analysis helps identify the customer characteristics and service-related factors that contribute most significantly to churn predictions.

### 5.10 Model Performance Comparison

A comparative analysis was performed between Random Forest and XGBoost using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

A grouped bar chart was generated to visualize the performance differences between the two models.

---

## 6. Model Performance

The following table presents the test-set evaluation results of both classification models.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 76.93% | 55.85% | 62.57% | 59.02% | 81.98% |
| XGBoost | To be updated | To be updated | To be updated | To be updated | To be updated |

### Random Forest Evaluation

Random Forest achieved a ROC-AUC score of approximately 81.98%, demonstrating its ability to distinguish between churn and non-churn customers.

The model achieved a recall of approximately 62.57% for the churn class, indicating that it successfully identified a considerable proportion of customers who actually churned.

The confusion matrix and classification report were further analyzed to understand the model's prediction performance across both classes.

*Note: XGBoost evaluation results and cross-validation scores will be updated after final model comparison.*

---

## 7. Key Findings

- The dataset contains 7,043 customer records and 20 independent features.
- Approximately 26.54% of customers belong to the churn class.
- The dataset exhibits class imbalance, with non-churn customers representing the majority class.
- Numerical and categorical preprocessing was implemented using Scikit-learn pipelines.
- Random Forest and XGBoost were used to develop churn prediction models.
- Class imbalance was addressed using class weights and scale_pos_weight.
- Multiple classification metrics were used for model evaluation.
- Stratified Cross-Validation was performed to assess model stability.
- Feature importance analysis was conducted to identify the most influential predictors.
- Model comparison was performed to evaluate the strengths and weaknesses of both classifiers.

---

## 8. Project Structure

The repository is organized as follows:

```text
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb
│
├── Telco-Customer-Churn.csv
│
├── README.md
│
└── requirements.txt
```

---

## 9. Installation and Execution

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd customer-churn-prediction
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

Open `customer_churn_prediction.ipynb` and execute the cells sequentially.

**Important:** Keep `Telco-Customer-Churn.csv` in the same directory as the notebook, or update the dataset path accordingly.

---

## 10. Requirements

The project dependencies are listed in `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

## 11. Skills Demonstrated

This project demonstrates practical knowledge and application of:

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Data Visualization
- Classification Algorithms
- Random Forest
- XGBoost
- Class Imbalance Handling
- Feature Encoding
- Machine Learning Pipelines
- Model Evaluation
- Cross-Validation
- ROC-AUC Analysis
- Confusion Matrix Interpretation
- Feature Importance
- Comparative Model Analysis

---

## 12. Conclusion

This project demonstrates an end-to-end machine learning classification workflow for predicting customer churn using the Telco Customer Churn Dataset.

Starting with exploratory data analysis and data preprocessing, the implementation progresses through class imbalance handling, model development, evaluation, cross-validation, and feature importance analysis.

Random Forest and XGBoost classifiers were developed and evaluated using multiple classification metrics. The models were further analyzed using confusion matrices, ROC curves, and cross-validation to understand their predictive performance and reliability.

The project highlights how machine learning can be applied to customer churn prediction and how predictive insights can support customer retention strategies, improve customer satisfaction, and reduce customer loss.

---

## 13. Acknowledgement

This project was developed as part of the **Datheon Machine Learning & Data Science program**, focusing on customer churn prediction using machine learning classification techniques.

---

*Machine Learning & Data Science Enthusiast*