import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

cities = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Ahmedabad", "Pune", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
    "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut"
]

# City base pollution profiles (multipliers relative to baseline)
city_profiles = {
    "Delhi":      {"pm25": 2.8, "pm10": 2.6, "no2": 2.4, "so2": 1.8, "co": 2.2, "o3": 0.8},
    "Mumbai":     {"pm25": 1.6, "pm10": 1.5, "no2": 1.8, "so2": 1.4, "co": 1.5, "o3": 1.0},
    "Kolkata":    {"pm25": 2.2, "pm10": 2.0, "no2": 1.9, "so2": 1.7, "co": 1.8, "o3": 0.9},
    "Chennai":    {"pm25": 1.3, "pm10": 1.4, "no2": 1.4, "so2": 1.2, "co": 1.3, "o3": 1.1},
    "Bangalore":  {"pm25": 1.1, "pm10": 1.2, "no2": 1.3, "so2": 1.0, "co": 1.1, "o3": 1.2},
    "Hyderabad":  {"pm25": 1.4, "pm10": 1.5, "no2": 1.4, "so2": 1.3, "co": 1.3, "o3": 1.0},
    "Ahmedabad":  {"pm25": 1.9, "pm10": 2.0, "no2": 1.6, "so2": 1.7, "co": 1.7, "o3": 0.9},
    "Pune":       {"pm25": 1.3, "pm10": 1.4, "no2": 1.3, "so2": 1.1, "co": 1.2, "o3": 1.1},
    "Jaipur":     {"pm25": 2.0, "pm10": 2.2, "no2": 1.5, "so2": 1.5, "co": 1.6, "o3": 0.9},
    "Lucknow":    {"pm25": 2.3, "pm10": 2.1, "no2": 1.8, "so2": 1.6, "co": 1.9, "o3": 0.8},
    "Kanpur":     {"pm25": 2.5, "pm10": 2.3, "no2": 2.0, "so2": 2.0, "co": 2.1, "o3": 0.8},
    "Nagpur":     {"pm25": 1.7, "pm10": 1.8, "no2": 1.5, "so2": 1.4, "co": 1.5, "o3": 1.0},
    "Indore":     {"pm25": 1.6, "pm10": 1.7, "no2": 1.4, "so2": 1.3, "co": 1.4, "o3": 1.0},
    "Bhopal":     {"pm25": 1.5, "pm10": 1.6, "no2": 1.3, "so2": 1.2, "co": 1.3, "o3": 1.1},
    "Patna":      {"pm25": 2.4, "pm10": 2.2, "no2": 1.9, "so2": 1.7, "co": 2.0, "o3": 0.8},
    "Ludhiana":   {"pm25": 2.1, "pm10": 2.0, "no2": 1.7, "so2": 1.6, "co": 1.8, "o3": 0.9},
    "Agra":       {"pm25": 2.2, "pm10": 2.1, "no2": 1.8, "so2": 1.6, "co": 1.9, "o3": 0.8},
    "Nashik":     {"pm25": 1.2, "pm10": 1.3, "no2": 1.2, "so2": 1.1, "co": 1.1, "o3": 1.1},
    "Faridabad":  {"pm25": 2.6, "pm10": 2.4, "no2": 2.2, "so2": 1.9, "co": 2.1, "o3": 0.7},
    "Meerut":     {"pm25": 2.3, "pm10": 2.1, "no2": 1.8, "so2": 1.7, "co": 1.9, "o3": 0.8},
}

start_date = datetime(2020, 1, 1)
end_date   = datetime(2024, 12, 31)
date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

rows = []
target_rows = 5000
per_city = target_rows // len(cities)

for city in cities:
    prof = city_profiles[city]
    sampled_dates = random.sample(date_range, per_city)
    sampled_dates.sort()

    for date in sampled_dates:
        month = date.month
        # Seasonal factor: winter (Nov-Feb) worse, monsoon (Jul-Sep) better
        if month in [11, 12, 1, 2]:
            season_factor = np.random.uniform(1.3, 1.7)
        elif month in [3, 4, 5]:
            season_factor = np.random.uniform(1.0, 1.3)
        elif month in [6, 7, 8, 9]:
            season_factor = np.random.uniform(0.6, 0.9)
        else:
            season_factor = np.random.uniform(0.9, 1.2)

        # Base weather
        if month in [12, 1, 2]:
            temp = np.random.normal(15, 6)
            humidity = np.random.normal(70, 12)
            wind_speed = np.random.normal(6, 2.5)
        elif month in [3, 4, 5]:
            temp = np.random.normal(32, 5)
            humidity = np.random.normal(40, 10)
            wind_speed = np.random.normal(10, 3)
        elif month in [6, 7, 8, 9]:
            temp = np.random.normal(30, 3)
            humidity = np.random.normal(80, 8)
            wind_speed = np.random.normal(14, 4)
        else:
            temp = np.random.normal(24, 5)
            humidity = np.random.normal(55, 10)
            wind_speed = np.random.normal(9, 3)

        wind_factor = max(0.3, 1.0 - (wind_speed - 5) * 0.04)
        humidity_factor = 1.0 + (humidity - 50) * 0.003

        noise = lambda std: np.random.normal(0, std)

        pm25 = max(5,  prof["pm25"] * season_factor * wind_factor * humidity_factor * 40  + noise(8))
        pm10 = max(10, prof["pm10"] * season_factor * wind_factor * humidity_factor * 70  + noise(15))
        no2  = max(5,  prof["no2"]  * season_factor * wind_factor * 30  + noise(6))
        so2  = max(2,  prof["so2"]  * season_factor * wind_factor * 15  + noise(4))
        co   = max(0.2,prof["co"]   * season_factor * wind_factor * 1.2 + noise(0.3))
        o3   = max(10, prof["o3"]   * (2.0 - season_factor * 0.5) * 40 + noise(8))

        # AQI computation (US EPA breakpoints approximation)
        def aqi_from_pm25(c):
            breakpoints = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                           (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
            for lo_c, hi_c, lo_i, hi_i in breakpoints:
                if lo_c <= c <= hi_c:
                    return round(((hi_i - lo_i)/(hi_c - lo_c)) * (c - lo_c) + lo_i)
            return 500

        def aqi_from_pm10(c):
            breakpoints = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
                           (255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]
            for lo_c, hi_c, lo_i, hi_i in breakpoints:
                if lo_c <= c <= hi_c:
                    return round(((hi_i - lo_i)/(hi_c - lo_c)) * (c - lo_c) + lo_i)
            return 500

        aqi = max(aqi_from_pm25(pm25), aqi_from_pm10(pm10))
        # Add contribution from other pollutants
        if no2 > 100: aqi = max(aqi, int(no2 * 1.5))
        if so2 > 75:  aqi = max(aqi, int(so2 * 1.8))
        aqi = min(500, max(0, aqi + int(noise(5))))

        rows.append({
            "city":        city,
            "date":        date.strftime("%Y-%m-%d"),
            "pm25":        round(pm25, 2),
            "pm10":        round(pm10, 2),
            "no2":         round(no2, 2),
            "so2":         round(so2, 2),
            "co":          round(co, 3),
            "o3":          round(o3, 2),
            "temperature": round(temp, 1),
            "humidity":    round(np.clip(humidity, 10, 100), 1),
            "wind_speed":  round(max(0.5, wind_speed), 1),
            "aqi":         aqi,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("air_quality_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df.head())
print(df.describe())
