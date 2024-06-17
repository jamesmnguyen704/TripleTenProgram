# Project Overview: Predicting Used Car Market Value for Rusty Bargain

## Objective
Rusty Bargain aims to develop an app that can predict the market value of a used car based on its technical specifications and trim versions. The goal is to compare models developed using gradient boosting libraries (LightGBM, XGBoost, and CatBoost) and Scikit-Learn's linear regression and random forest regression to determine the best-performing model in terms of prediction quality, training time, and prediction time.

## Data Description
- Features: 
  - date_crawled
  - price (target variable)
  - vehicle_type
  - registration_year
  - gearbox
  - power
  - model
  - mileage
  - registration_month
  - fuel_type
  - brand
  - not_repaired
  - date_created
  - number_of_pictures
  - postal_code
  - last_seen

## Table of Contents
1. Project Description
2. Data Preparation
3. Model Training and Evaluation
    - Linear Regression
    - Random Forest Regressor
    - LightGBM Regressor
    - CatBoost Regressor
4. Conclusion

## 1. Project Description
Rusty Bargain wants a model to predict the market value of a used car based on its specifications. The project involves developing models using gradient boosting libraries and Scikit-Learn's regression models to compare their performance.

## 2. Data Preparation
The data was loaded and cleaned by addressing missing values and outliers. Categorical features were encoded, and numerical features were scaled. The data was then split into training and testing sets.

## 3. Model Training and Evaluation
### Linear Regression
- RMSE: 2584
- Training Time: 22 CPU seconds
- Prediction Time: 3.5e-6 seconds per prediction

### Random Forest Regressor
- RMSE: 1550
- Training Time: 34.8 CPU seconds
- Prediction Time: 4.0e-5 seconds per prediction

### LightGBM Regressor
- RMSE: 1517
- Training Time: 1 minute 49 seconds
- Prediction Time: 7.8e-4 seconds per prediction

### CatBoost Regressor
- RMSE: 1516
- Training Time: 4 minutes 40 seconds
- Prediction Time: 1.3e-5 seconds per prediction

## 4. Conclusion
- Linear regression had the quickest training time but the highest RMSE.
- Random forest regression trained faster than gradient boosting models but had a higher RMSE.
- LightGBM and CatBoost showed similar RMSEs, with LightGBM having a slight edge in training speed.
- Overall, the LightGBM regression model is recommended for this task due to its balanced performance in terms of model quality and training time.
