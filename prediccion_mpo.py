import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import warnings

# Supresión de advertencias de ajuste del modelo
warnings.filterwarnings("ignore")

# Ruta a los datos combinados
data_folder = Path('data')
data_2016_path = data_folder / 'consolidated_2016.csv'
data_2024_path = data_folder / 'consolidated_2024.csv'

# Cargar los datos con configuración de codificación
data_2016 = pd.read_csv(data_2016_path, parse_dates=['FechaHora'], index_col='FechaHora', encoding='ISO-8859-1')
data_2024 = pd.read_csv(data_2024_path, parse_dates=['FechaHora'], index_col='FechaHora', encoding='ISO-8859-1')

# Asegurarse de que los índices de tiempo estén en frecuencia horaria
y = data_2016['MPO']
y.index = pd.date_range(start=y.index[0], end=y.index[-1], freq='H')

# Variables exógenas (asegurando que tengan la misma frecuencia de índice que y)
exog = data_2016.drop(columns=['MPO'])
exog = exog.resample('H').ffill().bfill()  # Rellena NaN y asegura frecuencia horaria

# División de datos para entrenamiento y prueba
train_end = int(len(y) * 0.8)
y_train, y_test = y[:train_end], y[train_end:]
exog_train, exog_test = exog[:train_end], exog[train_end:]

# Definir parámetros del modelo SARIMAX
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 24  # Ajustes de estacionalidad diaria

# Crear y ajustar el modelo SARIMAX
model = SARIMAX(y_train, exog=exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# Generar predicciones
pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=exog_test)

# Crear un DataFrame para comparar valores reales y predichos
predicciones = pd.DataFrame({'FechaHora': y_test.index, 'MPO_real': y_test, 'MPO_prediccion': pred})

# Guardar las predicciones en un archivo CSV
output_path = data_folder / 'predicciones_mpo.csv'
predicciones.to_csv(output_path, index=False)

print(f"Predicciones guardadas en: {output_path}")
