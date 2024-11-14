import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import pickle

# Definir carpeta de datos
data_folder = Path('data')

# Cargar los datos con el códec ISO-8859-1
data_2016 = pd.read_csv(data_folder / 'consolidated_2016.csv', parse_dates=['FechaHora'], index_col='FechaHora', encoding='ISO-8859-1')
data_2024 = pd.read_csv(data_folder / 'consolidated_2024.csv', parse_dates=['FechaHora'], index_col='FechaHora', encoding='ISO-8859-1')

# Concatenar datos y preparar variables endógenas y exógenas
data = pd.concat([data_2016, data_2024]).asfreq('h')  # Cambiar 'H' a 'h' para evitar la advertencia
y = data['MPO'].fillna(method='ffill')
exog = data.drop(columns=['MPO']).fillna(method='ffill').fillna(method='bfill')

# Parámetros del modelo SARIMAX
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 24

# Preparar la barra de progreso
tqdm_bar = tqdm(total=len(y), desc="Entrenando modelo SARIMAX", unit="iteración")

# Función de callback para actualizar la barra de progreso
def update_progress(params):
    tqdm_bar.update(1)

# Entrenar el modelo SARIMAX
model = SARIMAX(y, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(method='powell', disp=False)
tqdm_bar.close()

# Guardar el modelo entrenado
with open("sarimax_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)

print("Entrenamiento completado y modelo guardado.")
