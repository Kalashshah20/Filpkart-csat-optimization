#ML PART
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- ML PART (Corrected) ---
print("\n--- STARTING ML PART (Corrected) ---")

# Load data
df = pd.read_csv('/Users/labdhishah/Downloads/Project Files /Flipkart Project/Project Files/Flipkart Project/Customer_support_data.csv')

# --- ML Preprocessing ---
# Based on EDA, we select features that are mostly complete and relevant.
# 'connected_handling_time' is unusable (mostly null)
# 'customer_mood' does not exist.
# 'Agent_name', 'Supervisor', 'Manager' have too many unique values (high cardinality) 
# and would require more complex feature engineering.
# We will use the most robust, complete categorical features.

features_to_use = [
    'channel_name',
    'category',
    'Sub-category',
    'Tenure Bucket',
    'Agent Shift'
]

target_column = 'CSAT Score'

# Create a copy to avoid SettingWithCopyWarning
df_ml = df.copy()

# Fill NaNs in categorical features with 'Unknown'
for col in features_to_use:
    if df_ml[col].dtype == 'object':
        df_ml[col] = df_ml[col].fillna('Unknown')

# Create dummy variables for categorical features
X = pd.get_dummies(df_ml[features_to_use], drop_first=True)
y = df_ml[target_column]

# Save all feature names for later
feature_list = X.columns

# --- Build and Train the Model ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Initialize and Train the Model
# n_jobs=-1 uses all available processors
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining the Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate the Model ---
predictions = rf_model.predict(X_test)
print(f"\nModel R-squared (Accuracy): {r2_score(y_test, predictions):.2f}")
print(f"Model Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

# --- Identify Key Drivers ---
# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame of features and their importance scores
feature_importance_df = pd.DataFrame({
    'feature': feature_list,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Display the top 10 most important drivers
print("\n--- Top 10 Drivers of Customer Satisfaction ---")
print(feature_importance_df.head(10))

# Plot the importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), palette='rocket')
plt.title('Top 10 Drivers of Customer Satisfaction')
plt.savefig("top_10_drivers_ml.png")
print("Saved top_10_drivers_ml.png")
plt.clf()

print("\n--- ML PART COMPLETE ---")