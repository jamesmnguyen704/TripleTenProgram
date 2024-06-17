# Exploratory Data Analysis of Instacart

## Project Overview

This project involves performing an exploratory data analysis on the Instacart dataset. The dataset includes information about aisles, departments, orders, order products, and products.

## Project Structure

The project is structured as follows:

- `Instacart_EDA.ipynb`: The main Jupyter Notebook containing the analysis and results.
- `data/`: Directory containing the dataset files.
- `README.md`: This README file.
- `requirements.txt`: List of required Python libraries.

### Dataset Files

- `aisles.csv`
- `departments.csv`
- `instacart_orders.csv`
- `order_products.csv`
- `products.csv`

### Contents

1. **[Introduction](#introduction)**
2. **[Data Overview](#data-overview)**
3. **[Data Preprocessing](#data-preprocessing)**
4. **[Analysis](#analysis)**
5. **[Findings](#findings)**
6. **[Installation](#installation)**
7. **[Usage](#usage)**
8. **[Acknowledgments](#acknowledgments)**

## Introduction

In this project, we will perform an exploratory data analysis on the Instacart dataset to gain insights into customer purchasing behavior.

### Goals

1. Verify data quality and consistency.
2. Understand customer behavior patterns.
3. Identify popular products and shopping trends.

## Data Overview

The dataset includes the following files:

- `aisles.csv`: Contains information about the different aisles.
- `departments.csv`: Contains information about the different departments.
- `instacart_orders.csv`: Contains information about customer orders.
- `order_products.csv`: Contains information about products in each order.
- `products.csv`: Contains information about the products.

## Data Preprocessing

### Header Style

Ensure all column names are in snake_case and lowercase, and remove any spaces.

### Missing Values

Identify and handle missing values appropriately.

### Duplicates

Identify and remove duplicate values.

## Analysis

### Data Verification

- Verify that the `'order_hour_of_day'` values range from 0 to 23.
- Verify that the `'order_dow'` values range from 0 to 6.

### Customer Behavior

- Analyze the distribution of orders by hour of the day.
- Analyze the distribution of orders by day of the week.
- Analyze the distribution of waiting times between orders.

### Product Popularity

- Identify the top 20 most popular products.
- Analyze the distribution of the number of items per order.
- Identify the top 20 items that are reordered most frequently.

## Findings

### Data Verification

The data was found to be consistent, with `'order_hour_of_day'` values ranging from 0 to 23 and `'order_dow'` values ranging from 0 to 6.

### Customer Behavior

- Customers tend to place orders most frequently in the late morning to early afternoon.
- Orders are more frequent on certain days of the week, with notable peaks.
- The average waiting time between orders is identified.

### Product Popularity

- The top 20 most popular products were identified.
- The distribution of the number of items per order was analyzed.
- The top 20 most frequently reordered items were identified.

## Installation

To run this project, you will need the following libraries:

- pandas
- numpy
- matplotlib

You can install these libraries using pip:

```bash

