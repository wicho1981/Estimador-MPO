import pandas as pd
from pathlib import Path

# Ruta base para los archivos CSV
data_folder = Path('data')

# Cargar los datos de MPO con codificación ISO-8859-1
mpo_2016 = pd.read_csv(data_folder / 'dataset2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
mpo_2024 = pd.read_csv(data_folder / 'Dataset2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

# Transformar a formato largo (para MPO)
mpo_2016_long = mpo_2016.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO')
mpo_2016_long['Hora'] = mpo_2016_long['Hora'].astype(int)
mpo_2016_long['FechaHora'] = mpo_2016_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
mpo_2016_long.drop(columns=['Fecha', 'Hora'], inplace=True)

mpo_2024_long = mpo_2024.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO')
mpo_2024_long['Hora'] = mpo_2024_long['Hora'].astype(int)
mpo_2024_long['FechaHora'] = mpo_2024_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
mpo_2024_long.drop(columns=['Fecha', 'Hora'], inplace=True)

# Cargar los datos de caudal
caudal_2016 = pd.read_csv(data_folder / 'Aportes2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
caudal_2024 = pd.read_csv(data_folder / 'Aportes2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

# Promediar caudales por día, seleccionando solo columnas numéricas
caudal_2016_long = caudal_2016.groupby('Fecha').mean(numeric_only=True).reset_index()
caudal_2016_long = caudal_2016_long.reindex(caudal_2016_long.index.repeat(24)).reset_index(drop=True)
caudal_2016_long['Hora'] = caudal_2016_long.groupby('Fecha').cumcount()
caudal_2016_long['FechaHora'] = caudal_2016_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
caudal_2016_long.drop(columns=['Fecha', 'Hora'], inplace=True)

caudal_2024_long = caudal_2024.groupby('Fecha').mean(numeric_only=True).reset_index()
caudal_2024_long = caudal_2024_long.reindex(caudal_2024_long.index.repeat(24)).reset_index(drop=True)
caudal_2024_long['Hora'] = caudal_2024_long.groupby('Fecha').cumcount()
caudal_2024_long['FechaHora'] = caudal_2024_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
caudal_2024_long.drop(columns=['Fecha', 'Hora'], inplace=True)

# Cargar los datos de volumen de embalses (reservas)
volumen_2016 = pd.read_csv(data_folder / 'Reservas2016.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')
volumen_2024 = pd.read_csv(data_folder / 'Reservas2024.csv', delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

# Promediar volumen por día, seleccionando solo columnas numéricas
volumen_2016_long = volumen_2016.groupby('Fecha').mean(numeric_only=True).reset_index()
volumen_2016_long = volumen_2016_long.reindex(volumen_2016_long.index.repeat(24)).reset_index(drop=True)
volumen_2016_long['Hora'] = volumen_2016_long.groupby('Fecha').cumcount()
volumen_2016_long['FechaHora'] = volumen_2016_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
volumen_2016_long.drop(columns=['Fecha', 'Hora'], inplace=True)

volumen_2024_long = volumen_2024.groupby('Fecha').mean(numeric_only=True).reset_index()
volumen_2024_long = volumen_2024_long.reindex(volumen_2024_long.index.repeat(24)).reset_index(drop=True)
volumen_2024_long['Hora'] = volumen_2024_long.groupby('Fecha').cumcount()
volumen_2024_long['FechaHora'] = volumen_2024_long.apply(lambda row: row['Fecha'] + pd.Timedelta(hours=row['Hora']), axis=1)
volumen_2024_long.drop(columns=['Fecha', 'Hora'], inplace=True)

# Combinar los datasets
merged_2016 = mpo_2016_long.merge(caudal_2016_long, on='FechaHora', how='inner').merge(volumen_2016_long, on='FechaHora', how='inner')
merged_2024 = mpo_2024_long.merge(caudal_2024_long, on='FechaHora', how='inner').merge(volumen_2024_long, on='FechaHora', how='inner')

# Guardar los datasets combinados
output_2016_path = data_folder / 'consolidated_2016.csv'
output_2024_path = data_folder / 'consolidated_2024.csv'

merged_2016.to_csv(output_2016_path, index=False, encoding='ISO-8859-1')
merged_2024.to_csv(output_2024_path, index=False, encoding='ISO-8859-1')

print(f"Datos combinados guardados en:\n{output_2016_path}\n{output_2024_path}")
