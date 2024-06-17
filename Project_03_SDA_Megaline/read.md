# Telecom Plan Analysis: Surf vs. Ultimate

## Project Overview

As an analyst for the telecom operator Megaline, your task is to analyze the financial performance of two prepaid plans: Surf and Ultimate. The company aims to determine which plan generates more revenue to optimize the advertising budget. This project involves a comprehensive analysis of customer behavior and revenue generation based on data from 500 Megaline clients collected over a year.

## Goals

1. **Revenue Analysis**: Determine which prepaid plan, Surf or Ultimate, brings in more revenue.
2. **User Behavior Comparison**: Compare user behavior, including call durations, message counts, and internet usage, across the two plans.
3. **Statistical Hypothesis Testing**: Test statistical hypotheses to understand the differences in revenue between the two plans.

## Project Structure

The project is structured as follows:

- `Telecom_Plan_Analysis.ipynb`: The main Jupyter Notebook containing the analysis and results.
- `data/`: Directory containing the dataset files.
- `README.md`: This README file.
- `requirements.txt`: List of required Python libraries.

### Dataset Files

- `megaline_calls.csv`: Records of all calls made by users.
- `megaline_internet.csv`: Records of internet sessions by users.
- `megaline_messages.csv`: Records of text messages sent by users.
- `megaline_plans.csv`: Details of the Surf and Ultimate plans.
- `megaline_users.csv`: Information about the users, including their plan and registration details.

### Contents

1. **[Introduction](#introduction)**
2. **[Initialization](#initialization)**
3. **[Data Overview](#data-overview)**
4. **[Data Preprocessing](#data-preprocessing)**
5. **[Data Aggregation](#data-aggregation)**
6. **[User Behavior Analysis](#user-behavior-analysis)**
7. **[Revenue Analysis](#revenue-analysis)**
8. **[Statistical Hypothesis Testing](#statistical-hypothesis-testing)**
9. **[Conclusions](#conclusions)**
10. **[Installation](#installation)**
11. **[Usage](#usage)**
12. **[Acknowledgments](#acknowledgments)**

## Introduction

The purpose of this project is to analyze the revenue and user behavior of two prepaid plans, Surf and Ultimate, offered by the telecom operator Megaline. The analysis is based on data from 500 clients, exploring various aspects such as call durations, message counts, and internet usage to determine which plan is more profitable.

## Initialization

### Libraries

The following libraries are used in this project:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
