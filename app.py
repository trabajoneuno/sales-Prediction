from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import uvicorn

app = FastAPI(title="Sales Prediction API")

# Cargar datos y modelos al inicio
try:
    model = load_model('sales_prediction_model.h5')
    scaler = joblib.load('scaler.pkl')
    train_data = pd.read_csv('train.csv')
    stores_data = pd.read_csv('stores.csv')
    features_data = pd.read_csv('features.csv')
    
    # Convertir fechas a datetime
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    features_data['Date'] = pd.to_datetime(features_data['Date'])
    
except Exception as e:
    print(f"Error loading model and data: {str(e)}")
    raise

def prepare_sequence_for_prediction(store, dept, date, train_data, stores_data, features_data):
    """
    Prepara una secuencia para predicción sin escalado previo.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    store_info = stores_data[stores_data['Store'] == store].iloc[0]
    
    date_features = features_data[
        (features_data['Store'] == store) &
        (features_data['Date'] == date)
    ].iloc[0]

    store_sales = train_data[train_data['Store'] == store]['Weekly_Sales']
    dept_sales = train_data[
        (train_data['Store'] == store) &
        (train_data['Dept'] == dept)
    ]['Weekly_Sales']

    store_avg = store_sales.mean()
    store_std = store_sales.std()
    dept_avg = dept_sales.mean()
    dept_std = dept_sales.std()

    features = {
        'Store': store,
        'Dept': dept,
        'Temperature': date_features['Temperature'],
        'Fuel_Price': date_features['Fuel_Price'],
        'CPI': date_features['CPI'],
        'Unemployment': date_features['Unemployment'],
        'MarkDown1': date_features.get('MarkDown1', 0) if not pd.isna(date_features.get('MarkDown1')) else 0,
        'MarkDown2': date_features.get('MarkDown2', 0) if not pd.isna(date_features.get('MarkDown2')) else 0,
        'MarkDown3': date_features.get('MarkDown3', 0) if not pd.isna(date_features.get('MarkDown3')) else 0,
        'MarkDown4': date_features.get('MarkDown4', 0) if not pd.isna(date_features.get('MarkDown4')) else 0,
        'MarkDown5': date_features.get('MarkDown5', 0) if not pd.isna(date_features.get('MarkDown5')) else 0,
        'Month': date.month,
        'Week': date.isocalendar()[1],
        'DayOfWeek': date.dayofweek,
        'IsHoliday': int(date_features['IsHoliday']),
        'Size': store_info['Size'],
        'Store_Avg_Sales': store_avg,
        'Store_Std_Sales': store_std,
        'Dept_Avg_Sales': dept_avg,
        'Dept_Std_Sales': dept_std,
        'IsSummer': int(date.month in [6, 7, 8]),
        'IsWinter': int(date.month in [12, 1, 2]),
        'IsWeekend': int(date.dayofweek in [5, 6]),
        'Type_A': int(store_info['Type'] == 'A'),
        'Type_B': int(store_info['Type'] == 'B')
    }

    return pd.DataFrame([features])

def analyze_historical_patterns(train_data, store, dept, target_date):
    """
    Analiza patrones históricos con ajustes optimizados.
    """
    store_dept_data = train_data[
        (train_data['Store'] == store) &
        (train_data['Dept'] == dept)
    ].copy()

    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Patrones semanales con ventana móvil
    store_dept_data['DayOfWeek'] = store_dept_data['Date'].dt.dayofweek
    recent_data = store_dept_data[store_dept_data['Date'] <= target_date].tail(12)
    dow_pattern = recent_data.groupby('DayOfWeek')['Weekly_Sales'].mean()
    dow_factor = dow_pattern.get(target_date.dayofweek, 1.0) / dow_pattern.mean() if not dow_pattern.empty else 1.0

    # Patrones mensuales suavizados
    store_dept_data['Month'] = store_dept_data['Date'].dt.month
    month_pattern = store_dept_data.groupby('Month')['Weekly_Sales'].mean()
    month_factor = month_pattern.get(target_date.month, 1.0) / month_pattern.mean()
    month_factor = 1.0 + (month_factor - 1.0) * 0.5

    # Tendencia reciente ponderada
    recent_weeks = [4, 8, 12]
    weights = [0.5, 0.3, 0.2]
    recent_trends = []

    for weeks, weight in zip(recent_weeks, weights):
        data = store_dept_data[
            (store_dept_data['Date'] < target_date) &
            (store_dept_data['Date'] >= target_date - pd.Timedelta(weeks=weeks))
        ]
        if not data.empty:
            recent_trends.append((data['Weekly_Sales'].mean(), weight))

    if recent_trends:
        recent_trend = sum(trend * weight for trend, weight in recent_trends) / sum(weight for _, weight in recent_trends)
    else:
        recent_trend = store_dept_data['Weekly_Sales'].mean()

    # Volatilidad adaptativa
    recent_std = store_dept_data[
        store_dept_data['Date'] >= target_date - pd.Timedelta(weeks=12)
    ]['Weekly_Sales'].std()
    if pd.isna(recent_std):
        recent_std = store_dept_data['Weekly_Sales'].std()

    return {
        'dow_factor': dow_factor,
        'month_factor': month_factor,
        'recent_trend': recent_trend,
        'recent_std': recent_std
    }

class PredictionRequest(BaseModel):
    store: int
    dept: int
    date: str

class PredictionResponse(BaseModel):
    store: int
    department: int
    date: str
    predicted_sales: float
    details: dict

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Sales Prediction API",
        "endpoints": {
            "/predict": "POST - Make a sales prediction",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "data_loaded": {
            "train": train_data is not None,
            "stores": stores_data is not None,
            "features": features_data is not None
        }
    }

@app.post("/predict")
def predict_sales_endpoint(request: PredictionRequest):
    try:
        # 1. Analizar patrones históricos
        patterns = analyze_historical_patterns(train_data, request.store, request.dept, request.date)

        # 2. Obtener predicción base del modelo
        X_sequence = prepare_sequence_for_prediction(
            request.store, request.dept, request.date, train_data, stores_data, features_data
        )
        X_scaled = scaler.transform(X_sequence)
        base_prediction = model.predict(X_scaled, verbose=0)[0][0]

        # 3. Ajustar predicción con patrones históricos
        adjusted_prediction = base_prediction * (
            1.0 + (patterns['dow_factor'] - 1.0) * 0.7
        ) * (
            1.0 + (patterns['month_factor'] - 1.0) * 0.5
        )

        # 4. Combinar con tendencia reciente
        weight_model = 0.3
        weight_recent = 0.7
        final_prediction = (
            adjusted_prediction * weight_model +
            patterns['recent_trend'] * weight_recent
        )

        # 5. Aplicar límites adaptativos
        recent_mean = patterns['recent_trend']
        recent_std = patterns['recent_std']

        lower_bound = max(
            recent_mean - 1.5 * recent_std,
            train_data[
                (train_data['Store'] == request.store) &
                (train_data['Dept'] == request.dept)
            ]['Weekly_Sales'].min()
        )
        upper_bound = min(
            recent_mean + 1.5 * recent_std,
            train_data[
                (train_data['Store'] == request.store) &
                (train_data['Dept'] == request.dept)
            ]['Weekly_Sales'].max()
        )

        final_prediction = np.clip(final_prediction, lower_bound, upper_bound)

        # Preparar respuesta detallada
        details = {
            "base_prediction": float(base_prediction),
            "day_of_week_factor": float(patterns['dow_factor']),
            "month_factor": float(patterns['month_factor']),
            "recent_trend": float(patterns['recent_trend']),
            "adjusted_prediction": float(adjusted_prediction),
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        }

        return {
            "store": request.store,
            "department": request.dept,
            "date": request.date,
            "predicted_sales": float(final_prediction),
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
