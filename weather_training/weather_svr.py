###################################################

# Weather Support Vector Regressor Model Training Script
## This code does not save models/preprocessors to Azure Blob Storage directly.

###################################################

# Function Libraries.py
import joblib
import re
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient

# Data Manipulation
import pandas as pd
import numpy as np

# Pipeline for numerical features and categorical features
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model Selection
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Support Vector Regressor
from sklearn.svm import SVR

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Azure storage config (Replace with your actual connection string for testing)
account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_container_name = "models"
preprocessor_container_name = "preprocessors"
experiment_container_name = "experiment-tracking"

# Configure MLflow tracking
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "WeatherModelTraining")

experiment_name = experiment_name + "_SVR"

# Containers
data_container_name = "weather-data"
training_data_blob = "weather_data_20251028.csv"

# Load training data from Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(data_container_name)
data_client = container_client.get_blob_client(training_data_blob)
data = pd.read_csv(BytesIO(data_client.download_blob().readall()))

# Initialize MLflow client
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
artifact_location = f"wasbs://{experiment_container_name}@{account_name}.blob.core.windows.net"

# Set up MLflow experiment
if experiment is None:
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=artifact_location)
    print(f"Created new MLflow experiment '{experiment_name}' with artifact store in Azure Blob.")
else:
    experiment_id = experiment.experiment_id
    print(f"Using existing MLflow experiment '{experiment_name}' (ID: {experiment_id})")

try:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow connected to: {mlflow_tracking_uri}")
except Exception as e:
    print(f"MLflow connection failed: {e}")

# FEATURE ENGINEERING
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day

data["season"] = pd.cut(data['month'],
                        bins=[0, 2, 5, 8, 11, 12],
                        labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'],
                        right=True,
                        include_lowest=True,
                        ordered=False)

data.drop(columns=['date'], inplace=True)

# DATA PREPROCESSING
target = 'temperature_2m_mean'
categorical_cols = ['season']
numerical_cols = [col for col in data.columns if col not in categorical_cols + [target]]

numerical_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='mean')),
                            ('scaler', StandardScaler())])

categorical_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

preprocessor = ColumnTransformer([
                            ('num', numerical_pipeline, numerical_cols),
                            ('cat', categorical_pipeline, categorical_cols)])

y = data[target]
X = data.drop(columns=[target])

preprocessor.fit(X)
X = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())

# TRAIN/TEST SPLIT
ratio = 0.8
split_index = int(len(X) * ratio)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

cv = TimeSeriesSplit(n_splits=3)

svr_params = {
            'C': [1e1, 1e2, 1e3, 1e4],
            'epsilon': [1.0, 1.2, 1.5, 1.8],
            'gamma': ['scale']}

svr = SVR(kernel='linear')
svr_search = RandomizedSearchCV(estimator=svr,
                                param_distributions=svr_params,
                                n_iter=15,
                                scoring='neg_mean_absolute_error',
                                cv=cv,
                                n_jobs=-1,
                                random_state=1234,
                                verbose=1)

# MLflow Experiment Tracking
with mlflow.start_run():
    svr_search.fit(X_train, y_train)
    best_params = svr_search.best_params_
    mlflow.log_params(best_params)

    svr_pred = svr_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
    mae = mean_absolute_error(y_test, svr_pred)
    mape = mean_absolute_percentage_error(y_test, svr_pred)

    print(f"\n Model performance:")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  MAPE : {mape * 100:.2f}%")

    mlflow.log_metrics({"rmse": float(rmse),"mae": float(mae),"mape": float(mape)})

    # FINAL MODEL TRAINING
    model = SVR(**best_params)
    model.fit(X, y)

    # Log preprocessor and model to MLflow
    mlflow.sklearn.log_model(preprocessor, name="preprocessor")
    mlflow.sklearn.log_model(model, name="model")

    # Add metadata
    mlflow.set_tag("model_name", "weather_model_1.1.pkl")
    mlflow.set_tag("preprocessor", "preprocessor_1.1.pkl")

print("MLflow run completed and logged successfully!")