# Yandex.Music Data Analysis Project

## Project Overview

This project aims to analyze the music preferences of users in two cities, Springfield and Shelbyville, using real Yandex.Music data. The goal is to test three hypotheses:

1. User activity differs depending on the day of the week and from city to city.
2. On Monday mornings and Friday evenings, residents of Springfield and Shelbyville listen to different genres.
3. Springfield listeners prefer pop, while Shelbyville has more rap fans.

The project is divided into three main stages: Data Overview, Data Preprocessing, and Hypothesis Testing.

## Project Structure

The project is structured as follows:

- `Yandex_Music_Project.ipynb`: The main Jupyter Notebook containing the analysis and results.
- `data/`: Directory containing the dataset `music_project_en.csv`.
- `README.md`: This README file.
- `requirements.txt`: List of required Python libraries.

### Contents

1. **[Introduction](#introduction)**
2. **[Stage 1. Data Overview](#stage-1-data-overview)**
    - [Conclusions](#data-overview-conclusions)
3. **[Stage 2. Data Preprocessing](#stage-2-data-preprocessing)**
    - [2.1 Header Style](#21-header-style)
    - [2.2 Missing Values](#22-missing-values)
    - [2.3 Duplicates](#23-duplicates)
    - [Conclusions](#data-preprocessing-conclusions)
4. **[Stage 3. Hypothesis Testing](#stage-3-hypothesis-testing)**
    - [3.1 User Behavior in Two Cities](#31-user-behavior-in-two-cities)
    - [3.2 Music Preferences at the Beginning and End of the Week](#32-music-preferences-at-the-beginning-and-end-of-the-week)
    - [3.3 Genre Preferences in Springfield and Shelbyville](#33-genre-preferences-in-springfield-and-shelbyville)
5. **[Findings](#findings)**

## Introduction

Whenever we're doing research, we need to formulate hypotheses that we can then test. Sometimes we accept these hypotheses; other times, we reject them. To make the right decisions, a business must be able to understand whether or not it's making the right assumptions.

In this project, you'll compare the music preferences of the cities of Springfield and Shelbyville. You'll study real Yandex.Music data to test the hypotheses below and compare user behavior for these two cities.

### Goal

Test three hypotheses:
1. User activity differs depending on the day of the week and from city to city.
2. On Monday mornings, Springfield and Shelbyville residents listen to different genres. This is also true for Friday evenings.
3. Springfield and Shelbyville listeners have different preferences. In Springfield, they prefer pop, while Shelbyville has more rap fans.

### Stages

Data on user behavior is stored in the file `data/music_project_en.csv`. There is no information about the quality of the data, so you will need to explore it before testing the hypotheses.

Your project will consist of three stages:
1. Data overview
2. Data preprocessing
3. Testing the hypotheses

## Stage 1. Data Overview

Open the data on Yandex.Music and explore it. You'll need `pandas`, so import it and read the data.

### Data Overview Conclusions

Each row in the table stores data on a track that was played. Some columns describe the track itself: its title, artist, and genre. The rest convey information about the user: the city they come from, the time they played the track. It's clear that the data is sufficient to test the hypotheses. However, there are missing values.

## Stage 2. Data Preprocessing

Correct the formatting in the column headers and deal with the missing values. Then, check whether there are duplicates in the data.

### 2.1 Header Style

Ensure all column names are in snake_case and lowercase, and remove any spaces.

### 2.2 Missing Values

Fill in missing values with markers to avoid errors during analysis.

### 2.3 Duplicates

Remove both obvious and implicit duplicates to ensure data accuracy.

### Data Preprocessing Conclusions

We detected three issues with the data:
- Incorrect header styles
- Missing values
- Obvious and implicit duplicates

These issues were addressed to ensure the data is ready for analysis.

## Stage 3. Hypothesis Testing

### 3.1 User Behavior in Two Cities

Analyze user activity in Springfield and Shelbyville by counting the number of tracks played on specific days.

### 3.2 Music Preferences at the Beginning and End of the Week

Compare the top genres on Monday mornings and Friday evenings in both cities using a custom function.

### 3.3 Genre Preferences in Springfield and Shelbyville

Group the data by genre to determine the most popular genres in each city.

## Findings

We have tested the following three hypotheses:

1. User activity differs depending on the day of the week and from city to city.
2. On Monday mornings, Springfield and Shelbyville residents listen to different genres. This is also true for Friday evenings.
3. Springfield and Shelbyville listeners have different preferences. In Springfield, they prefer pop, while Shelbyville has more rap fans.

After analyzing the data, we concluded:

1. User activity in Springfield and Shelbyville depends on the day of the week, though the cities vary in different ways. The first hypothesis is fully accepted.
2. Musical preferences do not vary significantly over the course of the week in both Springfield and Shelbyville. We can see small differences in order on Mondays, but in both cities, people listen to pop music the most. So we can't accept this hypothesis. Missing values may have affected this result.
3. The musical preferences of users from Springfield and Shelbyville are quite similar. The third hypothesis is rejected. If there is any difference in preferences, it cannot be seen from this data.

## Installation

To run this project, you will need the following libraries:

- pandas
- numpy

You can install these libraries using pip:

```bash



