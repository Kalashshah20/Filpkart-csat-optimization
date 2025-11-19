import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

# --- EDA PART (Corrected) ---
print("--- STARTING EDA PART (Corrected) ---")

# Load the dataset
df = pd.read_csv('/Users/labdhishah/Downloads/Project Files /Flipkart Project/Project Files/Flipkart Project/Customer_support_data.csv')

# Display the first 5 rows
print("--- Data Head ---")
print(df.head())

# Get data types and non-null counts
print("\n--- Data Info ---")
df.info()

# Check for missing values
print("\n--- Missing Values (Sum) ---")
print(df.isnull().sum())

# Get statistics for numerical columns
# Note: 'connected_handling_time' is almost entirely null.
# We will use 'CSAT Score' and 'Item_price'.
print("\n--- Descriptive Statistics ---")
print(df[['CSAT Score', 'Item_price']].describe())

# --- Univariate Analysis ---
print("\n--- Generating Univariate Plots ---")

# 1. Distribution of CSAT Score
plt.figure(figsize=(8, 6))
sns.countplot(x='CSAT Score', data=df, palette='viridis')
plt.title('Distribution of Customer Satisfaction (CSAT) Scores')
plt.savefig("csat_score_distribution.png")
print("Saved csat_score_distribution.png")
plt.clf()

# 2. Distribution of Interaction Category
plt.figure(figsize=(10, 6))
sns.countplot(y='category', data=df, order = df['category'].value_counts().index, palette='Blues_d')
plt.title('Most Common Interaction Categories')
plt.savefig("category_distribution.png")
print("Saved category_distribution.png")
plt.clf()

# 3. Distribution of Support Channels
plt.figure(figsize=(8, 6))
sns.countplot(x='channel_name', data=df, palette='Spectral')
plt.title('Customer Contacts by Channel')
plt.savefig("channel_distribution.png")
print("Saved channel_distribution.png")
plt.clf()

# 4. Distribution of Agent Shift
plt.figure(figsize=(10, 6))
sns.countplot(x='Agent Shift', data=df, order = df['Agent Shift'].value_counts().index)
plt.title('Interactions by Agent Shift')
plt.savefig("agent_shift_distribution.png")
print("Saved agent_shift_distribution.png")
plt.clf()

# --- Bivariate Analysis ---
print("\n--- Generating Bivariate Plots ---")

# 1. CSAT Score vs. Interaction Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='CSAT Score', data=df)
plt.title('CSAT Score by Interaction Category')
plt.xticks(rotation=45)
plt.savefig("csat_vs_category.png")
print("Saved csat_vs_category.png")
plt.clf()

# 2. CSAT Score vs. Channel Name
plt.figure(figsize=(10, 6))
sns.boxplot(x='channel_name', y='CSAT Score', data=df)
plt.title('CSAT Score by Channel')
plt.savefig("csat_vs_channel.png")
print("Saved csat_vs_channel.png")
plt.clf()

# 3. CSAT Score vs. Agent Shift
plt.figure(figsize=(10, 6))
sns.violinplot(x='Agent Shift', y='CSAT Score', data=df)
plt.title('CSAT Score vs. Agent Shift')
plt.savefig("csat_vs_agent_shift.png")
print("Saved csat_vs_agent_shift.png")
plt.clf()

# 4. Team/Agent Performance
plt.figure(figsize=(12, 8))
# Filter for agents with a reasonable number of interactions (e.g., > 50)
agent_counts = df['Agent_name'].value_counts()
agents_to_keep = agent_counts[agent_counts > 50].index
df_filtered_agents = df[df['Agent_name'].isin(agents_to_keep)]

agent_performance = df_filtered_agents.groupby('Agent_name')['CSAT Score'].mean().sort_values(ascending=False)
print("\n--- Top 10 Agents by Avg. CSAT Score (with >50 interactions) ---")
print(agent_performance.head(10))

# Plotting bottom 10 agents for impact analysis
agent_performance.tail(10).plot(kind='barh')
plt.title('Bottom 10 Agents by Avg CSAT Score (with >50 interactions)')
plt.savefig("bottom_10_agents.png")
print("Saved bottom_10_agents.png")
plt.clf()

print("\n--- EDA PART COMPLETE ---")