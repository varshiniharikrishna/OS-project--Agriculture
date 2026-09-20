"""
Configuration File for AgriEdge Intelligence Server.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")
TRANSLATIONS_DIR = os.path.join(BASE_DIR, "translations")

# Server settings
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", 5050))
DEBUG = False

# Adaptive Scheduler Formula Weights (Total = 1.0)
# PriorityScore = w1*DiseaseRisk + w2*Severity + w3*CropImportance + w4*WeatherRisk + w5*DeadlineUrgency + w6*ResourceUrgency
SCHEDULER_WEIGHTS = {
    "w1_disease_risk": 0.25,      # Impact of predicted/suspected disease threat
    "w2_severity": 0.20,          # Impact of disease severity level (Critical, High, Moderate, Low)
    "w3_crop_importance": 0.15,   # Economic/food security importance of crop
    "w4_weather_risk": 0.15,      # Environmental humidity and precipitation risk
    "w5_deadline_urgency": 0.15,  # Time remaining before deadline expiration
    "w6_resource_urgency": 0.10   # Resource matching & allocation fit
}

# PDF Knowledge Base RAG Documents
AGRICULTURE_PDFS = [
    os.path.join(BASE_DIR, "agriculture1.pdf"),
    os.path.join(BASE_DIR, "agriculture2.pdf"),
    os.path.join(BASE_DIR, "agriculture3.pdf"),
    os.path.join(BASE_DIR, "agriculture4.pdf")
]

# Open-Meteo Weather API settings
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Default environmental conditions
DEFAULT_WEATHER = {
    "location": "Thanjavur, Tamil Nadu",
    "latitude": 10.7870,
    "longitude": 79.1378,
    "temperature": 29.5,
    "humidity": 82.0,
    "rain_probability": 65.0,
    "precipitation_mm": 2.4,
    "wind_speed": 12.5,
    "weather_condition": "Humid / Light Rain",
    "fungal_disease_risk": "HIGH",
    "last_updated": "Just now",
    "is_live_api": False
}

