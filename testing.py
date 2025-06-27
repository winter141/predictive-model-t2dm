import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap

def xgboost(x_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    return model.fit(x_train, y_train)

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
print(feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgboost(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Partial Dependence Plots")
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_test,
    features=range(len(feature_names)),
    feature_names=feature_names,
    ax=ax,
)
plt.show()

