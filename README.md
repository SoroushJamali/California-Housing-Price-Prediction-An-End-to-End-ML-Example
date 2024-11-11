# Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd

# Load the California housing dataset as a DataFrame
# The dataset includes features related to California housing data, which are used as predictors for the target (housing prices).
housing = fetch_california_housing(as_frame=True).frame

# Define target and feature matrices
# The target variable 'MedHouseVal' (median house value) is separated from the feature set.
y = housing.pop("MedHouseVal")  # Target variable (median house value)
X = housing  # Feature variables

# Feature Engineering: Adding new feature - Ratio of average bedrooms to average rooms
# This engineered feature provides a normalized metric representing the density of bedrooms per room,
# which may better capture housing density than using raw 'AveRooms' or 'AveBedrms'.
X["AveBedrmsRatio"] = X["AveBedrms"] / X["AveRooms"]
X = X.drop(columns=['AveRooms', "AveBedrms"])  # Drop original columns to avoid multicollinearity

# Clustering based on location features to group similar areas
# Clustering the 'Latitude' and 'Longitude' features groups properties into regional clusters, 
# adding a categorical 'Location' feature to help capture potential location-based price trends.
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X[['Latitude', 'Longitude']])
X['Location'] = kmeans.labels_  # Add cluster labels as a new feature
X = X.drop(columns=['Latitude', 'Longitude'])  # Drop the latitude and longitude columns

# Outlier Detection: Remove outliers using IsolationForest
# Outliers can skew the model and lead to overfitting. We use IsolationForest to filter out outliers,
# retaining only inlier data points, which represent typical housing data patterns.
isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)
X, y = X[outlier_pred == 1], y[outlier_pred == 1]  # Keep only inliers

# Train-test split
# Splitting data into training and test sets ensures we can validate the model's generalizability.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a preprocessing and modeling pipeline
# This pipeline standardizes features and applies linear regression as a model.
# Pipelines ensure that transformations are consistently applied, reducing risk of data leakage.
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features to mean 0, variance 1 for better model performance
    ('estimator', LinearRegression())  # Linear regression as the predictive model
])

# Define grid search parameters for Linear Regression
# Grid search allows us to explore multiple parameter settings efficiently. 
# Here we tune 'fit_intercept' and 'positive' to find the best configuration for linear regression.
param_grid = {
    'estimator__fit_intercept': [True, False],  # Fit intercept affects how the model handles the intercept term
    'estimator__positive': [True, False]        # Positive constraint restricts coefficients to positive values
}

# Set up and perform Grid Search with cross-validation
# GridSearchCV performs an exhaustive search over the parameter grid, optimizing for the lowest mean squared error.
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Output best model parameters and test score
# Reporting the best score and parameters gives insight into model performance and the optimal settings found.
print(f"Best score (cross-validated negative MSE): {grid_search.best_score_:.3f}")
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model on the test set
# Test set performance reflects model generalizability. R² score indicates the percentage of variance explained by the model.
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set R^2 score: {test_score:.3f}")
