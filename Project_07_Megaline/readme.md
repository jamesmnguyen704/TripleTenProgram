# Megaline Plan Recommendation Model

## Summary

This project involves developing a model to recommend one of Megaline's new plans (Smart or Ultra) based on customer behavior data. The goal is to create a model with at least 75% accuracy.

### Key Findings

1. **Decision Tree Classifier**:
   - Best model achieved an accuracy of 78% with a tree depth of 3.
   - Overfitting observed with deeper trees.

2. **Random Forest Classifier**:
   - Best model achieved an accuracy of 79% with 13 trees and a depth of 2.
   - No overfitting observed, making it the most optimal model.

3. **Logistic Regression**:
   - Model achieved an accuracy of 76%.
   - No overfitting observed.

4. **Model Quality**:
   - The RandomForestClassifier model achieved 77% accuracy on the test dataset.
   - Sanity check confirmed the model performs significantly better than random guessing.

### Conclusion

The RandomForestClassifier model with 13 trees and a depth of 2 is the best model for predicting the type of plan (Ultra or Smart) that users need. The model meets the accuracy threshold and passes the sanity check, making it reliable for production use.

### Code Summary

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
users = pd.read_csv('/datasets/users_behavior.csv')

# Data Preprocessing
users.drop_duplicates(inplace=True)

# Data Segmentation
features = users.drop('is_ultra', axis=1)
target = users['is_ultra']
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.40, random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(features_valid, target_valid, test_size=0.50, random_state=12345)

# Decision Tree Classifier
best_est = 0
best_score = 0
for depth in range(1, 6):
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model.fit(features_train, target_train)
    score = model.score(features_valid, target_valid)
    if score > best_score:
        best_score = score
        best_est = depth

# Random Forest Classifier
best_score = 0
best_tree = 0
best_depth = 0
for tree in range(10, 21):
    for depth in range(1, 4):
        model = RandomForestClassifier(random_state=12345, n_estimators=tree, max_depth=depth)
        model.fit(features_train, target_train)
        score = model.score(features_valid, target_valid)
        if score > best_score:
            best_score = score
            best_tree = tree
            best_depth = depth

# Logistic Regression
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
score_valid = model.score(features_valid, target_valid)

# Best Model Evaluation
best_model = RandomForestClassifier(random_state=12345, n_estimators=13, max_depth=2)
best_model.fit(features_train, target_train)
score_test = accuracy_score(target_test, best_model.predict(features_test))

# Sanity Check
np.random.seed(12345)
random_test = pd.Series(np.random.choice([0, 1], size=len(target_test)))
accuracy_random = accuracy_score(target_test, random_test)

print(f'RandomForestClassifier Test Accuracy: {score_test}')
print(f'Random Predictions Accuracy: {accuracy_random}')
