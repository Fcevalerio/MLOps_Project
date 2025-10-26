# Basic Libraries
import logging
import azure.functions as func

# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Date Libraries
from datetime import datetime

# Web Scraping Libraries
import requests_cache
from retry_requests import retry

# Custom Libraries
import openmeteo_requests

app = func.FunctionApp()

# Timer Triggered Function to Get Historical Weather Data Monthly
@app.timer_trigger(schedule="0 0 0 1 * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 

def Get_Weather_Data(myTimer: func.TimerRequest) -> None:

    logging.info('Running weather data update...')

    # Dynamically compute dates
    start_date = datetime(2020, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        logging.info(f'Python timer trigger function ran at {datetime.now().isoformat()}')
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 40.4165,
            "longitude": -3.7026,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "shortwave_radiation_sum", "cloud_cover_mean", "dew_point_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "surface_pressure_mean", "wind_gusts_10m_mean", "wet_bulb_temperature_2m_mean", "daylight_duration", "sunshine_duration", "snowfall_sum", "rain_sum"],
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        logging.info(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        logging.info(f"Elevation: {response.Elevation()} m asl")
        logging.info(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_shortwave_radiation_sum = daily.Variables(1).ValuesAsNumpy()
        daily_cloud_cover_mean = daily.Variables(2).ValuesAsNumpy()
        daily_dew_point_2m_mean = daily.Variables(3).ValuesAsNumpy()
        daily_relative_humidity_2m_mean = daily.Variables(4).ValuesAsNumpy()
        daily_pressure_msl_mean = daily.Variables(5).ValuesAsNumpy()
        daily_surface_pressure_mean = daily.Variables(6).ValuesAsNumpy()
        daily_wind_gusts_10m_mean = daily.Variables(7).ValuesAsNumpy()
        daily_wet_bulb_temperature_2m_mean = daily.Variables(8).ValuesAsNumpy()
        daily_daylight_duration = daily.Variables(9).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(10).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(11).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(12).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}

        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
        daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
        daily_data["dew_point_2m_mean"] = daily_dew_point_2m_mean
        daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
        daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
        daily_data["surface_pressure_mean"] = daily_surface_pressure_mean
        daily_data["wind_gusts_10m_mean"] = daily_wind_gusts_10m_mean
        daily_data["wet_bulb_temperature_2m_mean"] = daily_wet_bulb_temperature_2m_mean
        daily_data["daylight_duration"] = daily_daylight_duration
        daily_data["sunshine_duration"] = daily_sunshine_duration
        daily_data["snowfall_sum"] = daily_snowfall_sum
        daily_data["rain_sum"] = daily_rain_sum

        daily_dataframe = pd.DataFrame(data = daily_data)

        logging.info(f"First 5 rows of data:\n{daily_dataframe.head()}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

#print("Function app is running.")
#weather_data = Get_Weather_Data(myTimer=None)
#print(weather_data.head())