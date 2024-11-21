import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Función para cargar datos
def cargar_datos():
    """
    Carga el dataset y prepara la serie temporal para el modelo ARIMA.
    """
    # Cargar el archivo CSV con el delimitador correcto
    data = pd.read_csv("data/combined_data_2016.csv", delimiter=',', encoding='ISO-8859-1')
    
    # Verifica si 'FechaHora' está presente
    if 'FechaHora' not in data.columns:
        raise ValueError("La columna 'FechaHora' no se encuentra en el archivo CSV.")
    
    # Convertir 'FechaHora' en formato datetime y establecerla como índice
    data['FechaHora'] = pd.to_datetime(data['FechaHora'])
    data.set_index('FechaHora', inplace=True)
    
    # Seleccionar la serie temporal de la variable objetivo
    target = data["MPO"]
    
    return target

# Entrenar Modelo ARIMA
def entrenar_arima(series, order=(1, 1, 1)):
    """
    Entrena un modelo ARIMA en la serie temporal proporcionada.
    """
    print("Entrenando el modelo ARIMA...")
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

# Generar Predicciones y Evaluar Modelo
def evaluar_arima(model_fit, series, steps=30):
    """
    Genera predicciones y calcula las métricas de evaluación.
    """
    print("\nGenerando predicciones...")
    pred = model_fit.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()
    
    # Preparar datos para evaluación
    actual = series[-steps:]
    predictions = pred_mean[:steps]
    
    # Calcular métricas
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    
    print("\n=== Métricas de Evaluación ===")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2:.4f}")
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Predicciones": predictions,
        "Confianza": pred_ci
    }

# Generar Predicciones para 2025
def generar_predicciones_2025(model_fit, steps=365*24):
    """
    Genera predicciones para todo el año 2025 y las guarda en un archivo CSV.
    """
    print("\nGenerando predicciones para el año 2025...")
    pred = model_fit.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    
    # Crear un rango de fechas para 2025
    start_date = datetime(2025, 1, 1, 0)
    date_range = [start_date + timedelta(hours=i) for i in range(steps)]
    
    # Crear DataFrame de resultados
    predicciones_df = pd.DataFrame({
        "FechaHora": date_range,
        "Prediccion_MPO": pred_mean
    })
    
    # Guardar en archivo CSV
    output_file = "data/predicciones_mpo_2025.csv"
    predicciones_df.to_csv(output_file, index=False)
    print(f"Predicciones para 2025 guardadas en '{output_file}'.")

    return predicciones_df

# Graficar Resultados
def graficar_resultados(series, pred_mean, pred_ci, steps=30):
    """
    Grafica las predicciones junto con la serie temporal real y sus intervalos de confianza.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series[-100:], label='Serie Real', color='blue')
    plt.plot(pred_mean, label='Predicciones', color='red')
    plt.fill_between(
        pred_ci.index, 
        pred_ci.iloc[:, 0], 
        pred_ci.iloc[:, 1], 
        color='pink', alpha=0.3, label='Intervalo de Confianza'
    )
    plt.title("Predicciones del Modelo ARIMA")
    plt.xlabel("Fecha")
    plt.ylabel("MPO")
    plt.legend()
    plt.show()

# Ejecución Principal
def main():
    # Cargar los datos
    serie = cargar_datos()

    # Entrenar el modelo ARIMA
    arima_model = entrenar_arima(serie, order=(1, 1, 1))  # Ajusta el orden ARIMA si es necesario

    # Evaluar el modelo y calcular métricas
    metricas = evaluar_arima(arima_model, serie, steps=30)

    # Graficar resultados de evaluación
    graficar_resultados(serie, metricas["Predicciones"], metricas["Confianza"])

    # Generar predicciones para el año 2025
    predicciones_2025 = generar_predicciones_2025(arima_model)

# Ejecutar el script
if __name__ == "__main__":
    main()
