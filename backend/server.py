from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from geopy.geocoders import Nominatim
from functools import lru_cache
import joblib
import pandas as pd
import numpy as np
import pvlib
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

geolocator = Nominatim(user_agent="BIPV_Digital_Twin/1.0")

# Load your custom AI model
try:
    model = joblib.load('bipv_model.pkl')
    print("SUCCESS: AI Brain loaded into memory.")
except Exception as e:
    print(f"WARNING: Could not load model. Error: {e}")
    model = None

@lru_cache(maxsize=200)
def get_coordinates(city_name: str):
    try:
        location = geolocator.geocode(city_name)
        if location: return location.latitude, location.longitude
        return None, None
    except:
        raise HTTPException(status_code=503, detail="Map service error.")

# Fetch Typical Meteorological Year
def fetch_tmy_data(lat, lon):
    try:
        pvgis_response = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)
        tmy_data = pvgis_response[0]

        df = tmy_data[['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']].copy()
        df = df.fillna(0).clip(lower=0)
        return df
    except Exception as e:
        raise ValueError(f"Could not fetch TMY data: {e}")

# BIPV - Types of PV panels
BIPV_CATALOG = {
    "Standard Panel": {"default_eff": 0.20, "color": "#1c2331"},
    "Gold Glass": {"default_eff": 0.12, "color": "#D4AF37"},
    "Blue Glass": {"default_eff": 0.15, "color": "#0000FF"}
}

# --- UPGRADED SCHEMAS FOR MULTI-FACE CONFIGURATION ---

class FaceData(BaseModel):
    num_panels: int
    panel_area_m2: float
    material_id: str
    efficiency: Optional[float] = None
    surface_tilt: float
    surface_azimuth: float

class FullBuildingRequest(BaseModel):
    city_name: str
    faces: Dict[str, FaceData] # React sends {"south": {...}, "roof": {...}}

# --- THE MASTER INFERENCE ENDPOINT ---

@app.post("/predict-energy")
def predict_energy(data: FullBuildingRequest):
    if model is None: raise HTTPException(status_code=500, detail="AI Model missing.")
    
    # Step 1: Location
    lat, lon = get_coordinates(data.city_name)
    if not lat: raise HTTPException(status_code=404, detail="City not found.")
    
    try:
        # Step 2: Fetch weather data ONCE for the whole city
        df_base = fetch_tmy_data(lat, lon)
        site = pvlib.location.Location(lat, lon, tz='UTC')
        solar_position = site.get_solarposition(df_base.index)
        
        # Step 3: Calculate Global Weather Triggers (for your UI particles)
        daylight_hours = df_base[df_base['ghi'] > 0]
        total_daylight = len(daylight_hours)
        
        if total_daylight > 0:
            snowy_count = len(daylight_hours[daylight_hours['temp_air'] < 0])
            sunny_count = len(daylight_hours[(daylight_hours['dni'] > daylight_hours['dhi']) & (daylight_hours['temp_air'] >= 0)])
            cloudy_count = total_daylight - snowy_count - sunny_count
            
            p_sunny = int((sunny_count / total_daylight) * 100)
            p_snowy = int((snowy_count / total_daylight) * 100)
            p_cloudy = int((cloudy_count / total_daylight) * 100)
        else:
            p_sunny, p_snowy, p_cloudy = 100, 0, 0

        # Step 4: Initialize Master Totals
        total_yearly_kwh = 0
        combined_monthly = {month: 0 for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
        face_breakdown = {}

        # Step 5: Loop through every configured wall/roof
        for face_name, face in data.faces.items():
            if face.num_panels == 0 or face.material_id == 'none':
                continue # Skip walls without solar panels

            # Make a copy of the weather data just for this wall
            df_face = df_base.copy()

            # Angle of incidence specific to THIS wall's tilt and azimuth
            df_face['aoi'] = pvlib.irradiance.aoi(
                surface_tilt=face.surface_tilt, 
                surface_azimuth=face.surface_azimuth, 
                solar_zenith=solar_position['apparent_zenith'], 
                solar_azimuth=solar_position['azimuth']
            )
            
            df_face['hour'] = df_face.index.hour
            df_face['month'] = df_face.index.month
            df_face['surface_tilt'] = face.surface_tilt
            df_face['surface_azimuth'] = face.surface_azimuth
            
            # Predict using your XGBoost model
            features = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'aoi', 'surface_tilt', 'surface_azimuth', 'hour', 'month']
            df_face['predicted_w_m2'] = model.predict(df_face[features])
            
            # Determine efficiency 
            if face.efficiency is not None:
                final_eff = face.efficiency
            else:
                material_data = BIPV_CATALOG.get(face.material_id, {"default_eff": 0.15})
                final_eff = material_data["default_eff"]

            # Watts to Kilowatts
            total_area = face.num_panels * face.panel_area_m2
            df_face['final_kw'] = (df_face['predicted_w_m2'] * total_area * final_eff) / 1000.0
            
            # Add to Yearly Total
            face_yearly_total = int(df_face['final_kw'].sum())
            total_yearly_kwh += face_yearly_total
            face_breakdown[face_name] = face_yearly_total

            # Add to Monthly Totals
            monthly_kwh = df_face.groupby('month')['final_kw'].sum().round(0).astype(int)
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for i in range(1, 13):
                combined_monthly[month_names[i-1]] += int(monthly_kwh.get(i, 0))

        # Step 6: Return your exact schema
        return {
            "city": data.city_name,
            "yearly_total_kwh": total_yearly_kwh,
            "monthly_averages": combined_monthly,
            "face_breakdown": face_breakdown, # Extra data showing output per wall
            "ui_weather_triggers": {
                "percentage_sunny": p_sunny,
                "percentage_cloudy": p_cloudy,
                "percentage_snowy": p_snowy
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))