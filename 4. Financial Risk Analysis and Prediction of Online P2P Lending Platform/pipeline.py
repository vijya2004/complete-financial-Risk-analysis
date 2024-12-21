from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib

class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, classifiers, regressors):
        self.classifiers = classifiers
        self.regressors = regressors
        self.scaler = StandardScaler()
        self.encoders = {}
        self.categorical_cols = []

    def fit(self, X, y_class, y_reg):
        # Identify categorical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Fit LabelEncoder for each categorical column
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        self.fitted_classifiers = []
        self.fitted_regressors = []

        # Fit classifiers
        for clf in self.classifiers:
            clf.fit(X_scaled, y_class)
            self.fitted_classifiers.append(clf)

        # Fit regressors on filtered data
        for reg in self.regressors:
            X_filtered = X_scaled[y_class == 1]
            y_reg_filtered = y_reg[y_class == 1]
            reg.fit(X_filtered, y_reg_filtered)
            self.fitted_regressors.append(reg)

        return self

    def predict(self, X):
        # Encode categorical features
        X = X.copy()  # Avoid changing the original dataframe
        for col in self.categorical_cols:
            if col in self.encoders:
                le = self.encoders[col]
                X[col] = le.transform(X[col].astype(str))
            else:
                raise ValueError(f"Column '{col}' was not seen during fitting.")

        # Scale the features
        X_scaled = self.scaler.transform(X)

        # Predict class
        y_class_preds = np.array([clf.predict(X_scaled) for clf in self.fitted_classifiers])
        y_class_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=y_class_preds)  # Majority voting

        # Predict regression
        X_filtered = X_scaled[y_class_pred == 1]
        if len(X_filtered) == 0:
            return y_class_pred, np.array([])  # No samples for regression

        y_reg_preds = np.array([reg.predict(X_filtered) for reg in self.fitted_regressors])
        y_reg_pred = y_reg_preds.mean(axis=0)  # Averaging for regression

        return y_class_pred, y_reg_pred

# Load the dataset
df = pd.read_csv('assets/model_data.csv')

# Identify categorical columns
categorical_cols = ['LoanOriginationQuarter', 'EmploymentStatus']

# Split data into features and targets
X = df.drop(columns=['LoanStatus', 'ELA'])
y_class = df['LoanStatus']
y_reg = df['ELA']

# Split into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Initialize models
rfc = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)
lr = LinearRegression()
lasso = Lasso(random_state=42, max_iter=10000, alpha=1.0)

classifiers = [rfc, xgb]
regressors = [lr, lasso]

# Create and fit the custom pipeline
pipeline = CustomPipeline(classifiers=classifiers, regressors=regressors)
pipeline.fit(X_train, y_class_train, y_reg_train)

# Save the pipeline using joblib
joblib.dump(pipeline, 'assets/combined_pipeline.pkl')
