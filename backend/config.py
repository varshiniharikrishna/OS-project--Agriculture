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
PORT = 5000
DEBUG = False

# Default environmental conditions
DEFAULT_WEATHER = {
    "temperature": 29.5,
    "humidity": 82.0,
    "rain_probability": 65.0,
    "wind_speed": 12.5,
    "fungal_disease_risk": "HIGH"
}
