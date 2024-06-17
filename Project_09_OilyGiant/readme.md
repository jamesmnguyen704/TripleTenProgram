# Beta Bank Churn Prediction

## Project Description

Beta Bank is concerned about customer churn and wants to develop a model to predict which customers are likely to stop using their services. The goal is to create a model with a high F1-score (at least 0.59) and compare it with the AUC-ROC score.

## Data

The dataset includes information about customer behavior and contract status.

### Features

- `RowNumber`: index in the data rows
- `CustomerId`
- `Surname`
- `CreditScore`: client's credit rating
- `Geography`: country of residence
- `Gender`
- `Age`
- `Tenure`: number of years the client has been with the bank
- `Balance`: current account balance
- `NumOfProducts`: number of products held by the client
- `HasCrCard`: binary variable indicating if the client has a credit card
- `IsActiveMember`: binary variable indicating if the client is an active member
- `EstimatedSalary`
- `Exited`: binary target variable indicating if the client has exited

## Methodology

1. **Data Preprocessing**:
   - Check for missing values and duplicates.
   - Encode categorical variables.
   - Split the data into training, validation, and test sets.

2. **Model Training and Evaluation**:
   - Train various models (Decision Tree, Random Forest, Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes).
   - Evaluate models using F1-score and ROC-AUC score.
   - Tune hyperparameters for the RandomForestClassifier.

3. **Addressing Class Imbalance**:
   - Implement upsampling and downsampling techniques.
   - Evaluate the impact of these techniques on model performance.

4. **Final Model Selection**:
   - Select the best model based on F1-score, accuracy, and ROC-AUC score.
   - Perform a sanity check to ensure the model performs better than random chance.

## Data Loading and Initial Inspection

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Load dataset
data = pd.read_csv('/datasets/Churn.csv')

# Check for missing values
data['Tenure'].fillna(0, inplace=True)

# Check for duplicates
data.drop_duplicates(inplace=True)

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'])
data_encoded.drop(['Surname'], axis=1, inplace=True)

# Split data into features and target
target = data_encoded['Exited']
features = data_encoded.drop(['Exited'], axis=1)

# Split data into training, validation, and test sets
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.4, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(features_valid, target_valid, test_size=0.5, random_state=12345)
