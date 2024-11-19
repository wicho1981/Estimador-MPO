from pathlib import Path
import pandas as pd
import numpy as np

# Definir las rutas a los archivos de entrada y salida en la misma carpeta
dataset2016_path = Path(__file__).parent / 'dataset2016.csv'
reservas2016_path = Path(__file__).parent / 'Reservas2016.csv'
output_path = Path(__file__).parent / 'combined_data_2016.csv'

# Leer los datos de precios horarios (dataset2016)
mpo_data = pd.read_csv(dataset2016_path, delimiter=';', parse_dates=['Fecha'], dayfirst=True)

# Transformar los datos a formato largo (una fila por hora)
mpo_data_long = mpo_data.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO')
mpo_data_long['Hora'] = mpo_data_long['Hora'].astype(int)
mpo_data_long['FechaHora'] = mpo_data_long.apply(lambda x: x['Fecha'] + pd.Timedelta(hours=x['Hora']), axis=1)
mpo_data_long.drop(columns=['Fecha', 'Hora'], inplace=True)

# Leer los datos de reservas con codificación ISO-8859-1
reservas_data = pd.read_csv(reservas2016_path, delimiter=';', parse_dates=['Fecha'], dayfirst=True, encoding='ISO-8859-1')

# Expansión de los datos diarios de embalses a valores horarios con una variación del 5%
reservas_expanded = pd.DataFrame()

for index, row in reservas_data.iterrows():
    # Generar 24 variaciones para cada embalse
    day_variations = pd.DataFrame({col: row[col] * (1 + np.random.uniform(-0.05, 0.05, 24))
                                   for col in reservas_data.columns if col not in ['Fecha', 'Region Hidrologica', 'Nombre Embalse']})
    
    # Crear una columna 'FechaHora' con cada hora del día
    day_variations['FechaHora'] = [row['Fecha'] + pd.Timedelta(hours=i) for i in range(24)]
    day_variations['Nombre Embalse'] = row['Nombre Embalse']
    
    # Añadir los datos del día actual al DataFrame de expansión
    reservas_expanded = pd.concat([reservas_expanded, day_variations], ignore_index=True)

# Pivotear para tener una columna por embalse en el formato horario
reservas_expanded_pivot = reservas_expanded.pivot(index='FechaHora', columns='Nombre Embalse', values='Volumen util Diario %')
reservas_expanded_pivot.columns = ['Embalse_' + str(col) for col in reservas_expanded_pivot.columns]  # Renombrar columnas

# Combinar ambos datasets en un único DataFrame por FechaHora
combined_data = pd.merge(mpo_data_long, reservas_expanded_pivot, left_on='FechaHora', right_index=True, how='inner')

# Exportar el conjunto de datos combinado a un nuevo archivo CSV
combined_data.to_csv(output_path)

print(f"Archivo combinado guardado en: {output_path}")
