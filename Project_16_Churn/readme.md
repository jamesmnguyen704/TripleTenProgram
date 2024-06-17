# Predictive Churn Analysis for Telecom Services Using Machine Learning

## Project Description
This project aims to predict customer churn for a telecom operator by analyzing datasets related to contracts, personal details, and internet/phone service usage. Using exploratory data analysis (EDA), feature engineering, and boosting algorithms, we aim to achieve an AUC-ROC score of 0.88 or higher. The ultimate goal is to enable the telecom operator to proactively identify at-risk customers and deploy targeted retention strategies to minimize churn.

## Interconnect's Services
Interconnect provides two main types of services:
- **Landline Communication**: The telephone can be connected to several lines simultaneously.
- **Internet**: The network can be set up via a telephone line (DSL) or through a fiber optic cable.

Additional services include:
- **Internet Security**: Antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
- **Technical Support**: A dedicated technical support line (TechSupport)
- **Cloud Services**: Cloud file storage and data backup (OnlineBackup)
- **Streaming Services**: TV streaming (StreamingTV) and a movie directory (StreamingMovies)

Clients can choose either a monthly payment plan or sign a 1- or 2-year contract. They have various payment methods available and receive an electronic invoice after each transaction.

## Data Description
The data consists of files with contract information, personal data, internet services, and phone services, all linked by a unique customer ID:
- **contract.csv**: Contract information
- **personal.csv**: Client's personal data
- **internet.csv**: Information about internet services
- **phone.csv**: Information about telephone services

## Steps

### Data Cleaning and Feature Engineering
- Filled missing values and standardized column names.
- Converted necessary columns to the appropriate data types.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features (monthly charges, total charges, and tenure) using Min-Max scaling.
- Created binary churn targets and tenure features.

### Exploratory Data Analysis (EDA)
- Analyzed each feature in relation to churn using visualizations.
- Found that the majority of churn occurs within the first few months of service.
- Identified that higher monthly spenders are more likely to churn.
- Discovered that customers with additional services (e.g., online security) have lower churn rates.
- Observed that senior citizens have higher churn rates, while couples and families have lower churn rates.

### Model Building and Evaluation
- Used SMOTE to address class imbalance, resulting in balanced training data.
- Trained models using logistic regression, LGBM, and CatBoost.
- Hyperparameter tuning was performed using GridSearchCV.
- Best model: CatBoost classifier with an AUC-ROC score of 0.844.

### Final Model and Conclusion
- **Final model**: CatBoost classifier with learning_rate=0.01, max_depth=3, and n_estimators=1000.
- **Performance**: Achieved an AUC-ROC score of 0.844, indicating strong predictive power.
- **Key Insights**:
  - Customers are most likely to churn within the first few months.
  - Higher monthly charges correlate with higher churn rates.
  - Additional services, except streaming, tend to reduce churn.
  - Senior citizens are more likely to churn, whereas families and couples are less likely.

### Results and Business Recommendations
- **EDA Findings**: 
  - **Early churn**: A significant portion of churn occurs within the first 5 months of service.
  - **Monthly charges**: Higher monthly charges are associated with higher churn rates.
  - **Additional services**: Customers who subscribe to additional services like online security are less likely to churn.
  - **Demographic insights**: Senior citizens have a higher churn rate, while couples and families show lower churn rates.
- **Model Performance**: The CatBoost classifier achieved an AUC-ROC score of 0.844, demonstrating its effectiveness in predicting customer churn.
- **Business Recommendations**:
  - **Early intervention**: Implement aggressive promotions and discounts for new users to reduce early churn.
  - **Service bundling**: Encourage subscription to additional services (e.g., online security, tech support) to lower churn rates.
  - **Targeted retention**: Develop targeted retention strategies for senior citizens and high monthly spenders.
  - **Customer satisfaction**: Address potential dissatisfaction among high monthly spenders to prevent churn.

### Report Summary
- **Performed Steps**: All planned steps were completed, including data preprocessing, EDA, feature engineering, and model training. We addressed challenges in data preparation, particularly in scaling and encoding without data leakage.
- **Challenges and Solutions**: 
  - **Class imbalance**: Addressed using SMOTE to balance the training data.
  - **Data preparation**: Ensured no data leakage by careful handling of encoding and scaling.
- **Key Actions**: 
  - **EDA**: Uncovered important patterns in the data, such as the early churn trend.
  - **Data Preprocessing**: Properly encoded categorical variables and scaled numerical features.
  - **Model Training**: Used GridSearchCV for hyperparameter optimization and selected the best-performing model.

### Conclusion
This project successfully identified key factors contributing to customer churn and developed a predictive model that can help the telecom operator proactively retain at-risk customers. The insights derived from this analysis provide actionable strategies to improve customer retention and reduce churn rates. By focusing on early intervention, service bundling, and targeted retention efforts, the company can significantly enhance its customer retention strategies and minimize churn.
