# Causal-Aware Multimodal Demand Forecasting Using Time Series, Textual Sentiment, and Satellite Imagery

## Overview

This project implements a lightweight multimodal demand forecasting framework inspired by the CAMT (Causal-Aware Multimodal Transformer) concept. The objective is to improve retail demand forecasting by integrating multiple heterogeneous data sources instead of relying only on historical sales data.

The project combines:

* Historical retail sales data
* Customer review sentiment
* Satellite nightlight imagery

These modalities are temporally aligned and used for forecasting weekly sales using multiple machine learning and deep learning models.

---

# Problem Statement

Traditional demand forecasting models mainly depend on historical sales data and often ignore external factors such as customer sentiment and economic activity.

This project aims to:

* Integrate multimodal data sources
* Perform temporal alignment across heterogeneous datasets
* Analyze the contribution of each modality
* Compare classical and deep learning forecasting models

---

# Objectives

* Develop a multimodal demand forecasting framework
* Integrate sales, sentiment, and satellite-derived economic indicators
* Perform structured temporal alignment
* Implement and compare multiple forecasting models
* Evaluate forecasting performance using regression metrics

---

# Datasets Used

## 1. M5 Sales Dataset

Source: Walmart Retail Dataset (Kaggle)

Features used:

* Date
* Sales
* Category information

Dataset Characteristics:

* 1965 days of sales data
* 30,490 time series
* Multiple product categories

Purpose:
Used as the primary time-series forecasting dataset.

---

## 2. Amazon Reviews Dataset

Source: Amazon Reviews Dataset (Kaggle)

Features used:

* Review text
* Review time

Dataset Characteristics:

* ~650 MB raw dataset
* 568,454 reviews
* Filtered to 2014–2016 period

Purpose:
Used for extracting customer sentiment scores.

---

## 3. Thailand Nightlight Satellite Dataset

Source: VIIRS Night-Time Lights Dataset (Kaggle)

Features used:

* Year
* Satellite images

Dataset Characteristics:

* 10 years of satellite imagery
* Multiple city-level nightlight images

Purpose:
Used as a proxy for economic activity.

---

# Project Workflow

## Step 1: Data Collection

Three datasets were collected:

* Sales dataset
* Reviews dataset
* Satellite imagery dataset

All datasets were inspected and filtered to the common time period:

2014–2016

---

# Step 2: Data Preprocessing

## Sales Data Preprocessing

* Converted daily sales into weekly aggregated sales
* Cleaned missing values
* Formatted timestamps
* Created lag features

### Lag Feature

`sales_lag1` represents the previous week's sales and helps capture temporal dependency.

---

## Sentiment Data Preprocessing

Customer review text was processed using:

### VADER Sentiment Analysis

Each review received a compound sentiment score ranging from:

-1 → Very Negative
+1 → Very Positive

Weekly average sentiment scores were computed.

---

## Satellite Image Processing

Satellite images were:

* Converted to grayscale
* Converted into NumPy arrays
* Processed to compute average pixel brightness

### Average Brightness Formula

Average brightness was computed using:

`avg_brightness = img_array.mean()`

Purpose:
Brightness acts as a proxy for:

* Economic activity
* Urban development
* Human activity

---

# Step 3: Temporal Alignment

One major challenge was that datasets had different time granularities:

| Dataset   | Frequency         |
| --------- | ----------------- |
| Sales     | Daily             |
| Sentiment | Review timestamps |
| Satellite | Yearly            |

To solve this:

* Sales data was aggregated weekly
* Sentiment data was aggregated weekly
* Satellite yearly data was converted into weekly format using forward fill

All datasets were then merged into a unified multimodal dataframe.

---

# Step 4: Feature Engineering

## 1. Lag Features

Previous week sales were used as input features.

Purpose:
Capture temporal dependency in sales forecasting.

---

## 2. Feature Normalization

MinMax Scaling was applied.

Reason:
Different features had different ranges:

* Sales ≈ 200,000
* Sentiment ≈ 0 to 1
* Brightness ≈ small values

Normalization stabilizes deep learning training.

---

## 3. Sliding Window Technique

Sequential data was prepared using sliding windows.

Example:

* Weeks 1–8 → Predict Week 9
* Weeks 2–9 → Predict Week 10

Purpose:
Prepare sequential input for LSTM and GRU models.

---

# Models Implemented

## 1. Linear Regression

Used as the baseline forecasting model.

Purpose:
Provide a simple benchmark for comparison.

---

## 2. Polynomial Regression (Degree 2)

Polynomial regression extends linear regression using squared terms.

Equation:

`y = a + bx + cx²`

Purpose:
Capture non-linear relationships in sales data.

Result:
Polynomial Regression achieved the best overall performance in this project.

---

## 3. LSTM (Long Short-Term Memory)

LSTM is a recurrent neural network designed for sequential data.

Capabilities:

* Learns temporal dependencies
* Remembers long-term patterns
* Handles time-series forecasting effectively

Input:
Past 8 weeks of multimodal features

Output:
Next week sales prediction

---

## 4. GRU (Gated Recurrent Unit)

GRU is a simplified recurrent neural network similar to LSTM.

Key Components:

* Update gate
* Reset gate

Advantages:

* Fewer parameters
* Faster training
* Efficient sequence learning

GRU captured overall sales trends effectively but produced smoother predictions.

---

# Evaluation Metrics

Since this is a regression problem, the following metrics were used:

## 1. MAE (Mean Absolute Error)

Measures average prediction error.

Lower MAE = Better model

---

## 2. R² Score

Measures how well the model explains variation in sales data.

Higher R² = Better model fit

---

# Results

## MAE Comparison

| Model                 | MAE    |
| --------------------- | ------ |
| Linear Regression     | ~12934 |
| LSTM                  | ~12905 |
| GRU                   | ~12743 |
| Polynomial Regression | ~11267 |

---

## R² Score Comparison

| Model                 | R² Score |
| --------------------- | -------- |
| LSTM                  | ~0.20    |
| GRU                   | ~0.18    |
| Polynomial Regression | ~0.27    |

---

# Key Findings

* Polynomial Regression achieved the best performance
* Non-linear relationships exist in the dataset
* LSTM and GRU captured temporal dependencies effectively
* Multimodal integration improved forecasting performance
* Sentiment alone had weak correlation with sales
* Satellite brightness contributed as an economic proxy feature

---

# Exploratory Data Analysis

Performed analyses include:

* Central tendency analysis
* Dispersion analysis
* Correlation analysis
* Scatter plots
* Distribution analysis
* Skewness and kurtosis analysis

Observations:

* Sales showed moderate variability
* Sentiment distribution was positively skewed
* Weak direct correlation existed between sentiment and sales
* Multimodal modeling was justified due to complex relationships

---

# Novelty of the Project

Compared to the original CAMT paper:

* Implemented a lightweight practical framework
* Used LSTM and GRU instead of full Transformer architecture
* Performed multimodal temporal alignment
* Conducted ablation experiments
* Built a computationally efficient solution suitable for moderate-sized datasets

---

# Ablation Experiments

The contribution of each modality was evaluated by comparing:

* Sales only
* Sales + Sentiment
* Sales + Sentiment + Satellite

Purpose:
Understand the importance of each modality.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* NLTK VADER
* PIL (Python Imaging Library)

---

# Future Work

* Implement full Transformer-based CAMT architecture
* Use larger real-time datasets
* Integrate advanced NLP models such as BERT
* Perform region-specific satellite analysis
* Improve causal reasoning and multimodal fusion

---

# Conclusion

This project demonstrates that integrating multiple modalities such as sales data, textual sentiment, and satellite imagery can improve demand forecasting performance. Structured temporal alignment and non-linear modeling significantly enhanced prediction quality. The work also shows that lightweight recurrent and polynomial models can provide strong performance under moderate computational constraints.

---

# References

* CAMT: Causal-Aware Multimodal Transformer
* M5 Forecasting Dataset
* Amazon Reviews Dataset
* VIIRS Night-Time Lights Dataset
* TensorFlow/Keras Documentation
* Scikit-learn Documentation
