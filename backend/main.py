import os
import joblib
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# SETUP PATH & LOAD DATA
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "runner_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "nutrition.csv")

app = FastAPI(title="Warteg-Run AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    model, scaler = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
except: pass 

try:
    df_nutrisi = pd.read_csv(CSV_PATH)
    MENU_DATABASE = dict(zip(df_nutrisi['name'], df_nutrisi['calories']))
except:
    MENU_DATABASE = {}

def kmh_to_pace(speed_kmh):
    total_minutes = 60 / speed_kmh
    return f"{int(total_minutes)}:{int((total_minutes - int(total_minutes)) * 60):02d}/km"

class RunnerInput(BaseModel):
    makanan_dikonsumsi: List[str]
    weight: float
    age: int
    height: float

class TrainInput(BaseModel):
    model_type: str
    features: List[str]
    test_size: float

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'Calories Burned': np.random.randint(100, 1000, n_samples),
        'Weight(kg)': np.random.randint(50, 100, n_samples),
        'Running Speed(km/h)': np.random.uniform(5.0, 15.0, n_samples),
        'Age': np.random.randint(17, 60, n_samples),
        'Height(cm)': np.random.randint(150, 190, n_samples)
    })
    df['Time_Minutes'] = (df['Calories Burned'] * 200) / (df['Weight(kg)'] * 3.5 * df['Running Speed(km/h)'])
    df['Time_Minutes'] += np.random.normal(0, 2, n_samples)
    return df

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/menu")
def get_menu():
    # UPDATE: Sekarang API mengembalikan Array of Objects (Bukan cuma array of string)
    menu_list = [{"name": k, "calories": v} for k, v in MENU_DATABASE.items()]
    return {"menu_tersedia": menu_list}

@app.get("/eda_data")
def get_eda_data():
    df = generate_synthetic_data(500) 
    return df.to_dict(orient="list")

@app.post("/predict")
def predict_time(data: RunnerInput):
    total_kalori = sum([MENU_DATABASE[m] for m in data.makanan_dikonsumsi if m in MENU_DATABASE])
    if total_kalori == 0: return {"status": "error", "pesan": "Menu tidak valid."}

    tinggi_m = data.height / 100
    bmi = data.weight / (tinggi_m ** 2)
    
    if data.age > 50 or bmi >= 27.0:
        kategori_fisik = "Pemula / Overweight / Senior"
        speeds = [round(random.uniform(6.0, 7.5), 1), round(random.uniform(7.5, 8.5), 1)]
    elif data.age > 35 or bmi >= 24.0:
        kategori_fisik = "Menengah / Normal"
        speeds = [round(random.uniform(8.0, 9.5), 1), round(random.uniform(9.5, 11.0), 1)]
    else:
        kategori_fisik = "Atletis / Fit / Muda"
        speeds = [round(random.uniform(10.5, 12.0), 1), round(random.uniform(12.0, 14.0), 1)]

    hasil_rekomendasi = []
    for speed in speeds:
        input_df = pd.DataFrame([{'Calories Burned': total_kalori, 'Weight(kg)': data.weight, 'Running Speed(km/h)': speed, 'Age': data.age, 'Height(cm)': data.height}])
        prediction = model.predict(scaler.transform(input_df))
        hasil_rekomendasi.append({
            "opsi_pace": f"Pace {kmh_to_pace(speed)} ({speed} km/h)",
            "rekomendasi_waktu_menit": round(float(prediction[0]), 2)
        })
    
    return {"status": "success", "analisis_kesehatan": {"bmi_anda": round(bmi, 2), "kategori_fisik": kategori_fisik}, "ringkasan_makanan": {"total_kalori": total_kalori}, "rekomendasi": hasil_rekomendasi}

@app.post("/train")
def train_model_live(data: TrainInput):
    df_train = generate_synthetic_data(1000)
    X, y = df_train[data.features], df_train['Time_Minutes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data.test_size, random_state=42)
    
    live_scaler = StandardScaler()
    X_train_scaled = live_scaler.fit_transform(X_train)
    X_test_scaled = live_scaler.transform(X_test)
    
    live_model = RandomForestRegressor(n_estimators=100, random_state=42) if data.model_type == "Random Forest" else LinearRegression()
    live_model.fit(X_train_scaled, y_train)
    preds = live_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # UPDATE: Perbaikan BUG Classic Pandas Index vs Numpy Array 
    residuals = (y_test.values - preds)[:100].tolist() 
    
    importances = live_model.feature_importances_.tolist() if data.model_type == "Random Forest" else np.abs(live_model.coef_).tolist()

    return {
        "metrics": {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)},
        "features": data.features,
        "importances": importances,
        "scatter": {"y_true": y_test.values[:50].tolist(), "y_pred": preds[:50].tolist()},
        "residuals": residuals
    }