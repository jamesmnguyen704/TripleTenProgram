# Exploratory Data Analysis of Zuber Ride-Sharing Data

## Summary

This project involves the exploratory data analysis (EDA) of the Zuber ride-sharing data to understand passenger preferences and the impact of weather on ride frequency. The analysis includes data preprocessing, visualization, and hypothesis testing.

### Key Findings

1. **Top Companies**:
   - "Flash Cab" has the highest number of trips, indicating a large fleet and high demand.
   - Companies with "association" or "affiliation" in their names account for a significant portion of total trips.

2. **Top Drop-off Locations**:
   - "River North", "Streeterville", and "Loop" regions have the highest trip averages.
   - These areas are likely central business districts with high demand for rides.

3. **Weather Impact on Trip Duration**:
   - Trips are longer on bad weather days compared to good weather days.
   - The average duration of rides varies significantly between sunny and rainy days.

4. **Hypothesis Testing**:
   - There is a significant difference in the average duration of rides from the Loop to O'Hare International Airport on rainy Saturdays compared to non-rainy Saturdays.

### Conclusion

The analysis highlights the influence of weather conditions on ride duration and the importance of central business districts as high-demand areas. Understanding these patterns can help optimize fleet positioning and improve service efficiency. The hypothesis testing confirmed that weather significantly impacts ride duration.

### Code Summary

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as st

# Load datasets
company = pd.read_csv('/datasets/project_sql_result_01.csv')
location = pd.read_csv('/datasets/project_sql_result_04.csv')
weather = pd.read_csv('/datasets/project_sql_result_07.csv')

# Data Preprocessing
company.drop_duplicates(inplace=True)
weather.drop_duplicates(inplace=True)
weather = weather[weather['duration_seconds'] != 0]

# Visualization: Top 10 Companies by Trips
top10_company = company.sort_values(by='trips_amount', ascending=False).head(10)
sns.barplot(x='trips_amount', y='company_name', data=top10_company, palette='dark')
plt.title('Top 10 Company x Trips Amount')
plt.show()

# Visualization: Top 10 Locations by Average Trips
top10_location = location.sort_values(by='average_trips', ascending=False).head(10)
sns.barplot(x='average_trips', y='dropoff_location_name', data=top10_location, palette='dark')
plt.title('Top 10 Location Name x Average Trips')
plt.show()

# Boxplot: Trip Duration by Weather Conditions
sns.boxplot(x='weather_conditions', y='duration_seconds', data=weather, palette="dark")
plt.title('Boxplot of Trip Duration by Weather Conditions')
plt.show()

# Hypothesis Testing
df_good = weather[weather['weather_conditions'] == 'Good']
df_bad = weather[weather['weather_conditions'] == 'Bad']

t_stat, p_value = st.ttest_ind(df_good['duration_seconds'], df_bad['duration_seconds'])
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Conclusion
if p_value < 0.05:
    print("We reject the null hypothesis: The average duration of rides changes on rainy Saturdays compared to non-rainy Saturdays.")
else:
    print("We do not reject the null hypothesis: There is not enough evidence to conclude that the average duration of rides changes on rainy Saturdays.")

