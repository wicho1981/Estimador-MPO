import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Función para cargar y preparar datos
def cargar_datos():
    data = pd.read_csv("data/combined_data_2016.csv")
    selected_features = [col for col in data.columns if col.startswith("Embalse_")]
    features = data[selected_features]
    target = data["MPO"]

    # Escalado de características
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

    # Crear DataFrame escalado
    data_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    data_scaled['MPO'] = y_scaled.flatten()  # Asegurarse de que 'MPO' sea un vector unidimensional

    return data_scaled, scaler_X, scaler_y, selected_features

# Función para entrenar el modelo ARIMA
def entrenar_modelo(data_scaled, selected_features, order=(1,1,1)):
    # Dividir en conjunto de entrenamiento y validación
    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled.iloc[:train_size]
    val_data = data_scaled.iloc[train_size:]

    # Preparar variables
    exog_train = train_data[selected_features]
    endog_train = train_data['MPO']
    exog_val = val_data[selected_features]
    endog_val = val_data['MPO']

    # Entrenar el modelo ARIMA con variables exógenas
    model = SARIMAX(endog_train, exog=exog_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Validación
    predictions = model_fit.predict(start=train_size, end=len(data_scaled)-1, exog=exog_val)
    mse = ((predictions - endog_val) ** 2).mean()
    print(f"Mean Squared Error en validación: {mse}")

    # Guardar el modelo
    model_fit.save("data/arima_model.pkl")
    print("Modelo ARIMA guardado como 'arima_model.pkl'")
    return model_fit

# Generar predicciones anuales para 2025
def generar_predicciones_anuales(model_fit, scaler_X, scaler_y, selected_features, year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Simular datos para las características exógenas
    embalse_means = {feature: 0.5 for feature in selected_features}
    simulated_data = {
        "FechaHora": date_range,
        **{feature: [embalse_means[feature] * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(len(date_range))]
           for feature in selected_features}
    }
    simulated_df = pd.DataFrame(simulated_data)

    # Escalar las características
    X_future_scaled = scaler_X.transform(simulated_df[selected_features])

    # Realizar predicciones utilizando el método forecast
    predictions_scaled = model_fit.forecast(steps=len(X_future_scaled), exog=X_future_scaled)
    predictions_scaled = predictions_scaled.clip(0, 1)  # Asegurar que las predicciones estén en el rango [0,1]

    # Invertir el escalado de las predicciones
    y_future = scaler_y.inverse_transform(predictions_scaled.values.reshape(-1, 1))

    # Añadir las predicciones al DataFrame
    simulated_df["Prediccion_MPO"] = y_future.flatten()

    # Guardar las predicciones
    simulated_df.to_csv(f"data/predicciones_mpo_{year}.csv", index=False)
    print(f"Predicciones guardadas para el año {year} en 'data/predicciones_mpo_{year}.csv'.")

# Ejecución del flujo completo
data_scaled, scaler_X, scaler_y, selected_features = cargar_datos()
arima_model = entrenar_modelo(data_scaled, selected_features, order=(1,1,1))
generar_predicciones_anuales(arima_model, scaler_X, scaler_y, selected_features, 2025)
