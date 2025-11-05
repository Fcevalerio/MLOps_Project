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

def get_next_model_version(blob_service_client, base_name="weather_model"):

    "Checks existing models in the container and returns the next version string."

    container_client = blob_service_client.get_container_client(model_container_name)
    
    try:
        existing_blobs = list(container_client.list_blobs())
    except Exception as e:
        print(f"Warning: could not list blobs: {e}")
        existing_blobs = []

    versions = []
    pattern = re.compile(f"{base_name}_(\\d+\\.\\d+)\\.pkl")
    for blob in existing_blobs:
        match = pattern.match(blob.name)
        if match:
            versions.append(float(match.group(1)))
    
    if versions:
        next_version = max(versions) + 0.1
    else:
        next_version = 1.1  # first model
    
    return f"{base_name}_{next_version:.1f}.pkl"

def get_next_preprocessor_version(blob_service_client, base_name="preprocessor"):

    "Checks existing models in the container and returns the next version string."

    container_client = blob_service_client.get_container_client(preprocessor_container_name)
    
    try:
        existing_blobs = list(container_client.list_blobs())
    except Exception as e:
        print(f"Warning: could not list blobs: {e}")
        existing_blobs = []

    versions = []
    pattern = re.compile(f"{base_name}_(\\d+\\.\\d+)\\.pkl")
    for blob in existing_blobs:
        match = pattern.match(blob.name)
        if match:
            versions.append(float(match.group(1)))
    
    if versions:
        next_version = max(versions) + 0.1
    else:
        next_version = 1.1  # first model
    
    return f"{base_name}_{next_version:.1f}.pkl"

def retrain_model(data: pd.DataFrame):
    """Retrain the model, log metrics to MLflow, and save model/preprocessor to Azure Blob Storage with versioning."""
    
    # FEATURE ENGINEERING
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    data["season"] = pd.cut(
        data['month'],
        bins=[0, 2, 5, 8, 11, 12],
        labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'],
        right=True,
        include_lowest=True,
        ordered=False
    )

    data.drop(columns=['date'], inplace=True)

    # DATA PREPROCESSING
    target = 'temperature_2m_mean'
    categorical_cols = ['season']
    numerical_cols = [col for col in data.columns if col not in categorical_cols + [target]]

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    y = data[target]
    X = data.drop(columns=[target])

    preprocessor.fit(X)
    X = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())

    # SAVE PREPROCESSOR TO BLOB
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    preprocessor_name = get_next_preprocessor_version(blob_service_client)

    buffer = BytesIO()
    joblib.dump(preprocessor, buffer)
    buffer.seek(0)

    blob_client = blob_service_client.get_blob_client(preprocessor_container_name, preprocessor_name)
    blob_client.upload_blob(buffer, overwrite=True)

    print(f"Preprocessor uploaded as {preprocessor_name}")

    # TRAIN/TEST SPLIT
    ratio = 0.8
    split_index = int(len(X) * ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    cv = TimeSeriesSplit(n_splits=3)

    svr_params = {
        'C': [1e1, 1e2, 1e3, 1e4],
        'epsilon': [1.0, 1.2, 1.5, 1.8],
        'gamma': ['scale']
    }

    svr = SVR(kernel='linear')
    svr_search = RandomizedSearchCV(
        estimator=svr,
        param_distributions=svr_params,
        n_iter=15,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1,
        random_state=1234,
        verbose=1
    )

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

        # Save model artifact
        model_name = get_next_model_version(blob_service_client)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        blob_client = blob_service_client.get_blob_client(model_container_name, model_name)
        blob_client.upload_blob(buffer, overwrite=True)
        print(f"Model retrained and uploaded as {model_name}")

        # Log preprocessor and model to MLflow
        mlflow.sklearn.log_model(preprocessor, name="preprocessor")
        mlflow.sklearn.log_model(model, name="model")

        # Add metadata
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("preprocessor", preprocessor_name)

    print("MLflow run completed and logged successfully!")