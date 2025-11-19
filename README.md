# 🛒 E-Commerce Customer Satisfaction (CSAT) Intelligence Engine

## 📖 Project Overview

This project builds an end-to-end data science workflow to analyze customer support interactions and predict satisfaction scores. It performs deep Exploratory Data Analysis (EDA) on over 85,000 support tickets. It utilizes a Random Forest Regressor to identify the key drivers behind customer sentiment (CSAT), helping to optimize agent training and shift allocation.

## 🛠️ Tech Stack

* **Language:** Python (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Visualization:** Matplotlib, Seaborn
* **Data Source:** Customer Support Dataset (CSV)

## 📂 Files Description

* **`EDA PART.py`**: Python script that handles data cleaning, univariate/bivariate analysis, and generates visualization plots for agent and channel performance.
* **`#ML PART.py`**: The machine learning pipeline that performs One-Hot Encoding, splits training data, trains the Random Forest model, and calculates feature importance.
* **`Customer_support_data.csv`**: The source dataset containing support interaction logs.
* **`top_10_drivers_ml.png`**: Visual output showing the most critical factors (e.g., Return Requests) impacting customer satisfaction.
* **`csat_vs_agent_shift.png`**: Visualization comparing agent performance across Morning, Evening, and Split shifts.
* **`channel_distribution.png`**: Chart displaying the volume of interactions across Inbound, Outcall, and Email channels.

## 🚀 How to Run

1.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

2.  **Run EDA Pipeline**
    ```bash
    python "EDA PART.py"
    ```

3.  **Run Machine Learning Model**
    ```bash
    python "#ML PART.py"
    ```

## 📊 Key Insights

* **Driver Analysis:** Identified that **Return Requests** are the single highest predictor of customer satisfaction scores, outweighing agent tenure.
* **Channel Volume:** Visualized that **Inbound Calls** dominate support volume, but **Email** channels show higher variance in satisfaction.
* **Workforce Optimization:** Discovered performance gaps in **Split Shifts** compared to standard Morning/Evening shifts, suggesting actionable areas for workforce management.
