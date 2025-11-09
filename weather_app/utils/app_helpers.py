from azure.storage.blob import BlobServiceClient
from datetime import datetime
import re

def get_latest_weather_blob(blob_service_client, container_name: str):
    """
    Finds the latest 'weather_data_<date>.csv' file in the specified container.
    Returns (blob_name, blob_client)
    """
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs())

    pattern = re.compile(r"weather_data_(\d{4}-\d{2})\.csv")
    dated_blobs = []

    for blob in blobs:
        match = pattern.match(blob.name)
        if match:
            date_str = match.group(1)
            blob_date = datetime.strptime(date_str, "%Y-%m")
            dated_blobs.append((blob_date, blob.name))

    if not dated_blobs:
        raise ValueError("No weather_data_<date>.csv files found in container.")

    latest_blob = max(dated_blobs, key=lambda x: x[0])[1]
    print(f"Latest weather data blob found: {latest_blob}")
    return latest_blob

def list_weather_blobs(container_client):
    blobs = [blob.name for blob in container_client.list_blobs()]
    # Sort by date in filename, assuming format "weather_data_YYYYMMDD.csv"
    blobs.sort(key=lambda x: x.split("_")[-1].split(".")[0])
    return blobs