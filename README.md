# Exploratory Data Analysis Suite (EDA Suite)

A modular Python-based Exploratory Data Analysis (EDA) framework that streamlines data ingestion, cleaning, and visualization for rapid data profiling and anomaly detection.

This suite is designed to automate the repetitive parts of EDA — from fixing messy CSV headers to producing statistical summaries, boxplots, scatter plots, and correlation heatmaps — with minimal manual setup.

## Features

**Automatic Data Cleaning**

- Detects and drops unit rows or malformed headers

- Fixes Unnamed columns and assigns intuitive names

- Converts all numeric-looking columns to the correct types

- Converts date/time columns automatically

**Exploratory Analysis Tools**

- NaN and missing value reports

- Summary statistics (mean, standard deviation, quartiles, min/max)

- Outlier detection via IQR or Z-score

- Mixed data type detection

**Visual Analysis**

- Distribution plots

- Boxplots for single or multiple columns

- Scatter plots for paired variables

- Correlation heatmaps

**Modular Architecture**

- Built around dependency injection via EdaContainer

- Analyzer, profiler, and visualizer are dynamically assembled

- Easily extended for domain-specific analysis

Project Structure

<img width="704" height="244" alt="image" src="https://github.com/user-attachments/assets/f897c6af-aa78-4ea9-94f0-b92309587c3f" />

# How It Works

The main workflow follows four simple steps:

import pandas as pd

from eda_suite.containers import EdaContainer

## 1. Load the dataset
df = pd.read_csv("data.csv", delimiter=",", skiprows=23, encoding='latin1')

## 2. Clean the dataset
from my_cleaning_module import clean_dataframe
df = clean_dataframe(df)

## 3. Initialize the EDA container
container = EdaContainer()
container.config.df.from_value(df)

## 4. Run EDA
analyzer = container.analyzer()
analyzer.show_profile()
analyzer.show_descriptive_stats()
analyzer.show_missing_values()


**Refer to main.py for a more elaborate example**
