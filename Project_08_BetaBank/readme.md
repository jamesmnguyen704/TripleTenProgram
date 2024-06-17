# Beta Bank Churn Prediction

## Project Description

Beta Bank is concerned about customer churn and wants to develop a model to predict which customers are likely to stop using their services. The goal is to create a model with a high F1-score (at least 0.59) and compare it with the AUC-ROC score.

### Key Findings

1. **Data Preprocessing**:
   - No missing values or duplicate records.
   - Balanced data achieved through upsampling and downsampling.

2. **Model Training**:
   - Various models (Decision Tree, Random Forest, Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes) were tested.
   - Best model: Random Forest Classifier with an F1-score of 0.60 after upsampling.

3. **Evaluation Metrics**:
   - F1-score of the best model: 0.60 (above the threshold of 0.59).
   - Accuracy: 85%
   - ROC-AUC score: 0.73

### Conclusion

The RandomForestClassifier model with 161 trees and a max depth of 15 achieved the best performance with an F1-score of 0.60. This model exceeds the required F1-score threshold and provides reliable predictions for customer churn.

### Code Summary

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

# Load dataset
data = pd.read_csv('/datasets/Churn.csv')

# Data Preprocessing
data['Tenure'].fillna(0, inplace=True)
data.drop_duplicates(inplace=True)
data_encoded = pd.get_dummies(data, columns=['Geography', 'Gender'])
data_encoded.drop(['Surname'], axis=1, inplace=True)
target = data_encoded['Exited']
features = data_encoded.drop(['Exited'], axis=1)

# Data Splitting
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.4, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(features_valid, target_valid, test_size=0.5, random_state=12345)

# Upsampling function
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    return shuffle(features_upsampled, target_upsampled, random_state=12345)

# Upsampling
features_upsampled, target_upsampled = upsample(features_train, target_train, 10)

# Model Training and Evaluation
model = RandomForestClassifier(n_estimators=161, max_depth=15, random_state=12345)
model.fit(features_upsampled, target_upsampled)
predicted_y = model.predict(features_test)

# Evaluation Metrics
f1_value = f1_score(target_test, predicted_y)
acc_score = accuracy_score(target_test, predicted_y)
rocauc_score = roc_auc_score(target_test, predicted_y)

print(f'F1 Score: {f1_value}')
print(f'Accuracy Score: {acc_score}')
print(f'ROC-AUC Score: {rocauc_score}')

# Confusion Matrix
conf_matrix = sns.heatmap(metrics.confusion_matrix(target_test, predicted_y), annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title(f'Accuracy Score: {acc_score}', size=15)
plt.show()
