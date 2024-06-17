# Sure Tomorrow Insurance Project

## Objective
This project aims to address several tasks for the Sure Tomorrow insurance company using Machine Learning:
1. Identify customers who are similar to a given customer to assist the company's agents with marketing.
2. Predict whether a new customer is likely to receive an insurance benefit and compare the prediction model's performance against a dummy model.
3. Predict the number of insurance benefits a new customer is likely to receive using a linear regression model.
4. Protect clients' personal data without compromising the quality of the machine learning model.

## Libraries Used
- pandas
- numpy
- seaborn
- scikit-learn

## Data Preprocessing & Exploration

### Initialization
Libraries and modules required for the project were imported, and the dataset was loaded and checked for any obvious issues.

### Data Loading
The data was loaded and the columns were renamed for consistency. The dataset was then inspected for data types and descriptive statistics. The age column was converted to an integer type.

### Exploratory Data Analysis
A pair plot was used to check for potential groups of customers. It was observed that identifying obvious groups was challenging due to the complexity of multivariate distributions.

## Task 1: Similar Customers
A function was implemented to find k nearest neighbors for a given customer based on different distance metrics (Euclidean and Manhattan) and scaling conditions (scaled and non-scaled data). The function returns the k nearest neighbors for a specified customer.

Results:
- Without scaling, the Euclidean distance metric produced different results compared to the Manhattan distance metric.
- Scaling the data resulted in more consistent and meaningful neighbor selections, confirming the importance of scaling in the KNN algorithm.

## Task 2: Predicting Insurance Benefits
In this task, the problem was approached as a binary classification task. A KNN-based classifier was built and evaluated using the F1 metric for k values from 1 to 10, both on the original data and the scaled data. The performance of the KNN classifier was compared to that of a dummy model, which returns "1" with varying probabilities. The data was split into a 70:30 ratio for training and testing.

Results:
- The highest F1 score achieved with the KNN classifier on scaled data was 0.75 with k=5.
- The dummy model's F1 score varied based on the probability of predicting "1", with the highest score being 0.58 at P=0.5.

## Task 3: Linear Regression
A linear regression model was built to predict the number of insurance benefits a new customer is likely to receive. The model was implemented from scratch using matrix operations. The RMSE metric was used to evaluate the model's performance. The effect of data scaling on the model's performance was also assessed.

Results:
- RMSE without scaling: 1.24
- RMSE with scaling: 1.24
- The results indicated that scaling did not significantly impact the linear regression model's performance.

## Task 4: Data Obfuscation
Data obfuscation was performed by multiplying the feature matrix by an invertible matrix. It was shown both analytically and computationally that this transformation does not affect the performance of the linear regression model. The obfuscated data could be recovered using the inverse of the transformation matrix.

Results:
- RMSE for the obfuscated data: 1.24
- R-squared for the obfuscated data: 0.72
- The predicted values and quality metrics remained consistent before and after obfuscation.

## Conclusions
1. Data scaling is crucial for the KNN algorithm to ensure equitable contribution from all features to distance calculations.
2. The Manhattan distance metric was less influenced by data scaling, making it a dependable choice for KNN.
3. The linear regression model's performance remained consistent whether the features were scaled or not.
4. Data obfuscation using an invertible matrix did not affect the linear regression model's predicted values or quality metrics like RMSE and R-squared. This ensures that client data can be protected without compromising the model's performance.
