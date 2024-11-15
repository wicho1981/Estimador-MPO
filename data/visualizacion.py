# visualizacion.py

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from modeloAnfis import ANFIS  # Asegúrate de que ANFIS está bien definido en modeloAnfis.py

# Cargar los datos de prueba
data = pd.read_csv("data/combined_data_2016.csv")
selected_features = ["Embalse_AMANI", "Embalse_GUAVIO", "Embalse_PENOL", "Embalse_PLAYAS"]  # Usar solo las mismas cuatro características
X_test = data[selected_features].values
y_test = data["MPO"].values
dates = data["FechaHora"]  # Obtener las fechas para los resultados

# Escalar los datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_test_scaled = scaler_X.fit_transform(X_test)
y_test_scaled = scaler_y.fit_transform(y_test.reshape(-1, 1))

# Convertir a tensores
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Cargar el modelo
model_path = "data/anfis_model.pth"
anfis_model = ANFIS(n_inputs=len(selected_features), n_mf=2)  # Ajusta n_mf al valor que se usó en el entrenamiento

# Cargar los pesos con strict=False para evitar errores en caso de diferencias
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
anfis_model.load_state_dict(state_dict, strict=False)

# Realizar predicciones
with torch.no_grad():
    y_pred_scaled = anfis_model(X_test_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Desescalar a los valores originales

# Imprimir los resultados con las fechas
print("Fecha\t\tValor Real (MPO)\tPredicción (MPO)")
for date, real, pred in zip(dates, y_test, y_pred):
    print(f"{date}\t{real:.2f}\t\t{pred[0]:.2f}")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Valores Reales (MPO)", color="blue")
plt.plot(y_pred, label="Predicciones del Modelo ANFIS", color="red", linestyle="dashed")
plt.xlabel("Tiempo")
plt.ylabel("MPO")
plt.legend()
plt.title("Comparación de Valores Reales y Predichos")
plt.show()
