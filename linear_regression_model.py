import pandas as pd
from pathlib import Path

# Definir la carpeta donde están los datos
data_folder = Path(__file__).parent / 'data'

# Cargar archivos CSV con codificación ISO-8859-1
caudal_2016 = pd.read_csv(data_folder / 'Aportes2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
caudal_2024 = pd.read_csv(data_folder / 'Aportes2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

mpo_2016 = pd.read_csv(data_folder / 'dataset2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
mpo_2024 = pd.read_csv(data_folder / 'Dataset2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
mpo_calculado_2024 = pd.read_csv(data_folder / 'Datasetcalculado2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

precios_ideales_2016 = pd.read_csv(data_folder / 'preofe2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
precios_ideales_2024 = pd.read_csv(data_folder / 'preofe2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

volumen_2016 = pd.read_csv(data_folder / 'Reservas2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
volumen_2024 = pd.read_csv(data_folder / 'Reservas2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

# Mostrar las primeras filas de cada dataframe para confirmar la carga correcta (opcional)
print("Caudal 2016:\n", caudal_2016.head())
print("Caudal 2024:\n", caudal_2024.head())
print("MPO 2016:\n", mpo_2016.head())
print("MPO 2024:\n", mpo_2024.head())
print("MPO Calculado 2024:\n", mpo_calculado_2024.head())
print("Precios Ideales 2016:\n", precios_ideales_2016.head())
print("Precios Ideales 2024:\n", precios_ideales_2024.head())
print("Volumen 2016:\n", volumen_2016.head())
print("Volumen 2024:\n", volumen_2024.head())

# Aquí podrías continuar con la limpieza y preparación de datos o el análisis adicional
# Esto depende de cómo planeas integrar estos datos en el modelo.
