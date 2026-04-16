# Titanic Dataset EDA

A Python-based exploratory data analysis project on the Titanic dataset, completed as part of the Syntecxhub Data Science Internship (Week 3, Project 1).

## Overview

This project loads the Titanic dataset, inspects data quality, analyzes survival patterns across different passenger groups, and visualizes the findings using bar charts, boxplots, and violin plots.

## Features

- Data inspection: shape, dtypes, and missing value summary
- Survival rate analysis by sex, passenger class, and age bucket
- Visualizations: bar charts, boxplot, violin plot, and count plot
- Short insight report printed to the console

## Key Insights

- Females survived at a significantly higher rate than males (~74% vs ~19%)
- 1st class passengers had nearly double the survival rate of 3rd class passengers
- Children had the highest survival rate among all age groups
- Age distributions of survivors skew slightly younger across all classes
- 3rd class had the largest passenger count but the lowest survival proportion

## Project Structure

Titanic dataset EDA/
├── titanic_eda.py
├── titanic_eda_plots.png
├── requirements.txt
└── README.md

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python titanic_eda.py
```
