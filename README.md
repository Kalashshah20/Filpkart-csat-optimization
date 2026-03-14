🛒 Flipkart Customer Satisfaction (CSAT) Intelligence Engine
📖 Project Overview
This project delivers an end-to-end Machine Learning pipeline to analyze and predict customer satisfaction for Flipkart. By analyzing over 100,000 support interactions, we identifies the key drivers behind 5-star ratings and provides actionable insights to reduce customer churn.

The project is contained within a single, comprehensive Jupyter Notebook that covers everything from Data Wrangling and Statistical Hypothesis Testing to Advanced Ensemble Modeling.

🚀 Notebook Workflow
The project follows a structured Data Science Lifecycle:

Data Inspection & Cleaning: Handling 60%+ missing values and treating outliers in connected_handling_time using Winsorization.

Exploratory Data Analysis (EDA): 15+ visualizations covering Univariate, Bivariate, and Multivariate analysis.

Hypothesis Testing: Using T-Tests and Chi-Square tests to statistically validate feature importance.

Feature Engineering: One-Hot Encoding categorical variables and applying Log Transformations to skewed data.

Handling Imbalance: Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes.

Machine Learning: Comparative analysis between Random Forest, XGBoost, and Logistic Regression.

Model Tuning: Optimization via GridSearchCV to maximize F1-Score and Recall.

🛠️ Tech Stack
Language: Python 3.12

Libraries: Pandas, NumPy, Scipy (Stats), Matplotlib, Seaborn

Machine Learning: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)

📊 Key Business Insights
Experience Gap: Agent tenure is a critical predictor; "Veteran" agents (>90 days) achieve significantly higher 5-star rates.

Efficiency Paradox: Data shows that extremely long calls correlate with lower satisfaction, highlighting a need for better real-time support for complex queries.

Channel Strategy: Digital channels (Chat/Inbound) show higher efficiency and satisfaction potential compared to traditional Email.

📈 Final Model Results
Best Model: Tuned XGBoost

Focus Metric: Recall (ensuring we identify as many dissatisfied customers as possible for service recovery).
