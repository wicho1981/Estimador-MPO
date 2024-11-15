from pathlib import Path
import pandas as pd

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

# Transformar los datos de reservas a un formato de columna por embalse usando "Volumen util Diario %"
reservas_pivot = reservas_data.pivot(index='Fecha', columns='Nombre Embalse', values='Volumen util Diario %')
reservas_pivot.columns = ['Embalse_' + str(col) for col in reservas_pivot.columns]  # Renombrar columnas
reservas_pivot.index.name = 'Fecha'  # Asegurar que la columna de índice se llama 'Fecha'

# Combinar ambos datasets en un único DataFrame por Fecha
combined_data = pd.merge(mpo_data_long, reservas_pivot, left_on='FechaHora', right_index=True, how='inner')
combined_data.set_index('FechaHora', inplace=True)  # Configurar FechaHora como índice

# Exportar el conjunto de datos combinado a un nuevo archivo CSV
combined_data.to_csv(output_path)

print(f"Archivo combinado guardado en: {output_path}")
