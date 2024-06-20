# README

## Project Overview

This project involves building a predictive model to determine customer churn (whether a customer will exit or not). The dataset is divided into training and test sets. The primary goal is to build and evaluate multiple models, perform hyperparameter tuning, and make predictions on the test set.

## Dataset Description

The dataset contains customer information with the following features:

- `CustomerId`: Unique identifier for each customer
- `Surname`: Customer's surname
- `CreditScore`: Customer's credit score
- `Geography`: Country of the customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Tenure`: Number of years the customer has been with the bank
- `Balance`: Account balance of the customer
- `NumOfProducts`: Number of products the customer has purchased through the bank
- `HasCrCard`: Whether the customer has a credit card (1: yes, 0: no)
- `IsActiveMember`: Whether the customer is an active member (1: yes, 0: no)
- `EstimatedSalary`: Estimated salary of the customer
- `Exited`: Whether the customer exited the bank (1: yes, 0: no)

## Installation and Setup

To run this project, you need the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

Install the libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Project Structure

- `forth.py`: Contains code for demonstrating and visualizing four models for simple binary classification.
- `bank_churn.ipynb`: Contains code for feature engineering, model training, hyperparameter tuning, and evaluation.
- `bank_churn.py`: Convert bank_churn.ipynb to bank_churn.py.
- `test.py`: Visualize and display the experimental results report.
- `dataset/`: Directory containing the training and test datasets.

## Steps and Methodology

### 1. Data Exploration

The initial data exploration includes:

- Reading the datasets.
- Displaying sample data.
- Checking data types and summary statistics.
- Checking for missing values.

### 2. Exploratory Data Analysis (EDA)

#### 2.1 Null Values

Checked for null values in the training and test datasets.

#### 2.2 Target Variable Analysis

Analyzed the distribution of the target variable (`Exited`) using pie and count plots.

#### 2.3 Categorical Variables Analysis

Analyzed categorical variables using pie and count plots.

#### 2.4 Numerical Value Analysis

Analyzed numerical variables using histograms and box plots.

#### 2.5 Correlation Analysis

Generated a correlation matrix to understand relationships between numerical features.

#### 2.6 Outlier Detection

Used box plots to detect outliers in numerical features.

### 3. Modeling

#### 3.1 Data Preparation

- Split the data into features (`X`) and target variable (`y`).
- Further split the data into training and validation sets.
- Defined a preprocessor for scaling numerical features and one-hot encoding categorical features.

#### 3.2 Model Training and Hyperparameter Tuning

- Trained four different models: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
- Used `RandomizedSearchCV` for hyperparameter tuning and cross-validation.

#### 3.3 Model Evaluation

- Evaluated models using accuracy, classification reports, confusion matrices, and ROC curves.
- Compared model performance based on these metrics.

### 4. Submission

- Made predictions on the test set using the best-performing model.
- Created a submission file with the predicted values.

## Results

The project evaluates multiple models and selects the best-performing one based on various metrics. The results are visualized using confusion matrices and ROC curves, providing insights into model performance.

## Conclusion

This project provides a comprehensive approach to predicting customer churn using various machine learning techniques. By performing detailed EDA, feature engineering, and model tuning, the best model is identified and used for making predictions on new data.

## Usage

To run the project, execute the notebooks in the following order:
`data_exploration.ipynb`
Ensure that the dataset files are placed in the `dataset/` directory.

## Acknowledgements

This project is based on a customer churn dataset. Special thanks to the dataset providers and the open-source community for their tools and libraries.
