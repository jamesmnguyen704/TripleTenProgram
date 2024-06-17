# Video Game Sales Analysis

## Summary

This project involves the exploratory data analysis (EDA) of the `games.csv` dataset, focusing on video game sales across different platforms and regions. The analysis aims to provide insights into market trends, genre popularity, and regional differences. Key steps in the project include data preprocessing, visualization, and hypothesis testing to understand user preferences and sales patterns.

### Key Findings

1. **Game Release Trends**:
   - The number of games released fluctuates over the years, with notable peaks and declines potentially correlated with technological advancements and platform transitions.

2. **Sales Variations Across Platforms**:
   - Platforms such as PS2 and Wii show high total sales, indicating their popularity during their peak years.

3. **Platform Life Cycles**:
   - Analyzing yearly sales per platform reveals the rise and fall of each platform, with PS2 and Wii experiencing significant sales before declining.

4. **Global Sales Analysis**:
   - North America and Europe are the largest markets for video game sales, with notable peak sales around the mid-2000s to early 2010s.

5. **User and Critic Review Analysis**:
   - There is a weak correlation between user scores and total sales, suggesting other factors like brand loyalty and marketing efforts also play a significant role.
   - Critic scores show a slightly stronger correlation with total sales compared to user scores.

6. **Comparative Analysis by Genre and Platform**:
   - Certain genres perform better on specific platforms, and the distribution of total sales varies significantly across genres.

7. **Regional User Profiles**:
   - Sports and action genres are more popular in North America and Europe, while role-playing games are more successful in Japan.
   - The most popular platforms vary by region, reflecting different market preferences and trends.

8. **Hypothesis Testing**:
   - Hypothesis tests reveal no significant difference between the average user scores for Xbox One and PC games, or between action and sports genres.

### Conclusion

The video game industry is characterized by dynamic trends in platform dominance, genre popularity, and regional market preferences. Understanding these patterns is crucial for developers and marketers to make data-driven decisions that align with current market trends. While user scores provide some insight, other factors such as marketing and platform exclusivity significantly impact sales performance.

### Code Summary

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
games = pd.read_csv('/datasets/games.csv')

# Data Preprocessing
games.dropna(subset=['Year_of_Release', 'Genre', 'Name'], inplace=True)
games['Year_of_Release'] = games['Year_of_Release'].astype(int)
games['Rating'].fillna('Unknown', inplace=True)
games['Genre'] = games['Genre'].astype('category')
games['Platform'] = games['Platform'].astype('category')
games['Rating'] = games['Rating'].astype('category')
games['total_sales'] = games[['NA_sales', 'EU_sales', 'JP_sales', 'Other_sales']].sum(axis=1)

# Analysis: Sales Variations Across Platforms
platform_sales = games.groupby('Platform')['total_sales'].sum().sort_values(ascending=False)
platform_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales per Platform')
plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')
plt.xticks(rotation=45)
plt.show()

# Analysis: Global Sales Trends Over Time
yearly_sales = games.groupby('Year_of_Release').agg({
    'NA_sales': 'sum',
    'EU_sales': 'sum',
    'JP_sales': 'sum',
    'Other_sales': 'sum'
})
yearly_sales.plot(figsize=(12, 6))
plt.title('Global Sales Trends Over Time')
plt.xlabel('Year of Release')
plt.ylabel('Total Sales (in millions)')
plt.grid(True)
plt.show()

# Hypothesis Testing
relevant_years = [2014, 2015, 2016]
recent_games = games[games['Year_of_Release'].isin(relevant_years)]
xbox_one_ratings = recent_games[(recent_games['Platform'] == 'XOne') & recent_games['User_Score'].notna()]['User_Score']
pc_ratings = recent_games[(recent_games['Platform'] == 'PC') & recent_games['User_Score'].notna()]['User_Score']
t_stat, p_value = stats.ttest_ind(xbox_one_ratings, pc_ratings)
print(f'T-statistic for Platform: {t_stat}, P-value for Platform: {p_value}')

