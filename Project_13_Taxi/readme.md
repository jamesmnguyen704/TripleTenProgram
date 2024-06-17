# Project: Predicting Taxi Orders at Airports

## Project Description
Sweet Lift Taxi company has historical data on taxi orders at airports. The goal is to predict the number of taxi orders for the next hour to attract more drivers during peak hours. The model should achieve an RMSE metric on the test set of no more than 48.

## Table of Contents
1. Project Description
2. Data Description
3. Import Libraries and Preprocess the Data
4. Resample the Data by One Hour
5. Exploratory Data Analysis (EDA)
6. Train Different Models with Various Hyperparameters
7. Feature Engineering
8. Split the Data
9. Scaling the Features
10. Training Different Models
    - Linear Regression
    - Random Forest Regressor
    - CatBoost Regressor
    - LightGBM Regressor
11. Conclusion

## Summary

### Data Preparation
The data was imported from the given dataset and resampled to hourly intervals to align with the goal of predicting taxi orders for the next hour. Initial exploratory data analysis was performed to understand the data structure and identify patterns. This involved checking for missing values, analyzing distributions, and identifying outliers.

### Exploratory Data Analysis (EDA)
The data exhibited non-stationarity, necessitating a transformation to a stationary state for effective machine learning model training. Analysis through decomposition and box plots revealed August as the peak month, with Mondays and Fridays being the busiest days. Outlier removal was performed for orders exceeding 300. Lag plots, autocorrelation function (ACF), and partial autocorrelation function (PACF) analyses indicated correlations in recent lags, which informed the feature engineering process.

### Feature Engineering
Several features were engineered to enhance model performance:
- **Temporal Features**: Month, hour, and day of the week.
- **Lag Features**: Previous time steps' values.
- **Rolling Statistics**: Rolling mean, minimum, and maximum values.

The dataset was made stationary to ensure effective training, and missing values were dropped to maintain data integrity.

### Model Training
Several models were trained and evaluated using different algorithms and hyperparameters:

- **Linear Regression**: This model was trained as a baseline and achieved an RMSE of 2584 on the cross-validated data. Although it had the quickest training time, the RMSE was relatively high.
- **Random Forest Regressor**: Hyperparameter tuning was performed, and the model achieved an RMSE of 1550. This model showed significant improvement over the linear regression model.
- **CatBoost Regressor**: Initially, without hyperparameter tuning, the model achieved an RMSE of 1613. After tuning, the CatBoost model achieved an RMSE of 1516, showing competitive performance.
- **LightGBM Regressor**: This model achieved the best RMSE of 1517 after extensive hyperparameter tuning, indicating it was the most suitable model for the given task.

### Results
The LightGBM Regressor was identified as the best-performing model, with the lowest RMSE of 45.87, thus meeting the project goal of achieving an RMSE of less than 48. The model's training involved feature engineering, hyperparameter tuning, and scaling of features to ensure optimal performance.

### Conclusion
In this project, the goal was to predict the number of taxi orders for the next hour based on historical data using various machine learning models. The process involved thorough data preprocessing, including resampling, outlier removal, and feature engineering. Several models were trained and evaluated, with the LightGBM Regressor emerging as the best model, achieving an RMSE of 45.87. This project demonstrated effective techniques in data preparation, feature engineering, and model evaluation to achieve the desired prediction accuracy. The findings can be utilized to improve the operational efficiency of taxi services by anticipating demand and allocating r

