# BIPV---Digital-Twin
building implemented photovoltaic prediction model and website for 3d visualization (work in progress)

# BIPV Predictive Digital Twin

A full-stack predictive simulation engine for **Building-Integrated Photovoltaics (BIPV)**. 

This platform acts as a digital twin, combining physics-based solar modeling with machine learning to predict the exact energy yield of solar panels mounted on different building faces (walls, roofs). It ingests localized climate data, structural configurations, and material efficiencies to provide real-time solar yield forecasting.

## System Architecture & Logic

This project moves beyond standard API development by integrating geospatial physics with predictive machine learning:

* **Geospatial & Physics Engine (`pvlib`):** Ingests a city name, fetches the Typical Meteorological Year (TMY) climate data, and mathematically calculates the exact Angle of Incidence (AOI) based on the sun's position and the specific tilt/azimuth of the building's walls.
* **Predictive ML Pipeline (`XGBoost`):** Replaces heavy, slow physical simulations with a lightweight, pre-trained XGBoost Regressor (`bipv_model.pkl`). It takes 10 meteorological and structural features (GHI, DNI, DHI, Temperature, AOI, Tilt, Azimuth, etc.) and predicts the target power density (W/m²).
* **Inference API (`FastAPI`):** A high-performance Python backend that handles the routing, runs the inference matrix for multi-face building configurations, and returns aggregated monthly and yearly Kilowatt-hour (kWh) estimates.

## Tech Stack

**Backend & Data Science**
* **Framework:** FastAPI (Python)
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
* **Solar Physics:** `pvlib`, `geopy`
* **Deployment:** Uvicorn, Joblib (Model Serialization)

**Frontend (WIP)**
* **Framework:** React, Vite
* **3D Rendering:** Three.js, WebGL, `@react-three/drei`

## Repository Structure

* `/backend` — Contains the core FastAPI application (`server.py`) and the compiled machine learning weights (`bipv_model.pkl`).
* `train_model.py` — The data engineering and model training pipeline. Demonstrates how NASA earth observation data was fetched, cleaned, and used to train the XGBoost regressor.
* `/frontend_wip` — Contains the React/Three.js prototype (`Modular_kit.jsx`), demonstrating how the raw ML data will be visualized onto interactive 3D assets. (wip)

##  Core API Endpoint

### `POST /predict-energy`
Accepts a JSON payload detailing the building's location and the configuration of its solar faces.

**Sample Request:**
```json
{
  "city_name": "Oslo",
  "faces": {
    "south_wall": {
      "num_panels": 10,
      "panel_area_m2": 1.6,
      "material_id": "Standard Panel",
      "surface_tilt": 90.0,
      "surface_azimuth": 180.0
    }
  }
}
