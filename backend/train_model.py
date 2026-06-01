import requests
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import joblib
import pvlib

# basic cities 
Climate_zones = {
    "Nagpur" : {"lat": 21.1458, "lon": 79.0882},
    "Oslo": {"lat": 59.9139, "lon": 10.7522},
    "Cairo": {"lat": 30.0444, "lon": 31.2357}
}

# ingestion
def fetch_nasa_hourly_yearly(lat, lon, start_year, end_year):
    print(f"Fetching NASA data for Lat: {lat}, Lon: {lon}")
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    parameters = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M"
    yearly_data = []

    for year in range(start_year, end_year +1): # we add 1 to the year for the next year beginning
        url = f"{base_url}?parameters={parameters}&community=RE&longitude={lon}&latitude={lat}&start={year}0101&end={year}1231&format=JSON" 
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status() # BUG 1 FIXED: Added ()
            data = response.json()['properties']['parameter']
            yearly_data.append(pd.DataFrame(data))
        except Exception as e:
            print(f"Failed year {year}: {e}") # BUG 2 FIXED: Fixed f-string format
        time.sleep(1.5)

    if not yearly_data: return pd.DataFrame()

    master_df = pd.concat(yearly_data)
    master_df.index = pd.to_datetime(master_df.index, format='%Y%m%d%H').tz_localize('UTC')
    master_df = master_df.rename(columns={'ALLSKY_SFC_SW_DWN': 'ghi', 'ALLSKY_SFC_SW_DNI': 'dni', 'ALLSKY_SFC_SW_DIFF': 'dhi', 'T2M': 'temp_air', 'WS10M': 'wind_speed'})
    master_df = master_df.replace(-999.0, pd.NA).dropna()
    master_df[['ghi', 'dni', 'dhi']] = master_df[['ghi', 'dni', 'dhi']].clip(lower=0)
    return master_df

# physics for bipv
def apply_bipv_physics(df, lat, lon):
    print("Calculating AOI and true Plane-of-Array (POA) = ")
    site = pvlib.location.Location(lat, lon, tz='UTC')
    solar_position = site.get_solarposition(df.index)

    df['solar_zenith'] = solar_position['apparent_zenith']
    df['solar_azimuth'] = solar_position['azimuth']

    np.random.seed(42)
    df['surface_tilt'] = np.random.choice([0, 20, 30, 45, 60, 90], size=len(df))
    df['surface_azimuth'] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315], size=len(df))

    df['aoi'] = pvlib.irradiance.aoi(df['surface_tilt'], df['surface_azimuth'], df['solar_zenith'], df['solar_azimuth'])

    poa_data = pvlib.irradiance.get_total_irradiance(
        surface_tilt=df['surface_tilt'], surface_azimuth=df['surface_azimuth'],
        solar_zenith=df['solar_zenith'], solar_azimuth=df['solar_azimuth'],
        dni=df['dni'], ghi=df['ghi'], dhi=df['dhi']
    )

    df['target_power_density'] = poa_data['poa_global']
    df['hour'] = df.index.hour
    df['month'] = df.index.month

    return df.dropna()

# training
if __name__ == "__main__":
    print("In - Memory training pipeline")
    all_training_data = []

    START_YEAR, END_YEAR = 2023, 2023

    # BUG 3 FIXED: Changed ',' to 'in'
    for city, coords in Climate_zones.items():
        # BUG 4 FIXED: Indented everything inside the loop
        print(f"\nprocessing {city}")
        raw_df = fetch_nasa_hourly_yearly(coords['lat'], coords['lon'], START_YEAR, END_YEAR)
        if not raw_df.empty:
            physics_df = apply_bipv_physics(raw_df, coords['lat'], coords['lon'])
            all_training_data.append(physics_df)
    
    master_dataset = pd.concat(all_training_data)

    print("\nInitializing XGBoost Regressor - training")
    features = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'aoi', 'surface_tilt', 'surface_azimuth', 'hour', 'month']
    X = master_dataset[features]
    y = master_dataset['target_power_density']
    
    model = xgb.XGBRegressor(n_estimators = 100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    model.fit(X, y)

    print("saving")
    joblib.dump(model, 'bipv_model.pkl')
    print("created")