import pandas as pd
from pathlib import Path

# Ruta del archivo CSV
file_path = Path("data/Datasetcalculado2016.csv")

# Leer el archivo CSV, asegurando que la columna 'Fecha' se interprete como datetime
data = pd.read_csv(file_path, delimiter=';', parse_dates=['Fecha'], dayfirst=True)

# Verificar y convertir explícitamente la columna 'Fecha' a datetime
if not pd.api.types.is_datetime64_any_dtype(data['Fecha']):
    data['Fecha'] = pd.to_datetime(data['Fecha'], dayfirst=True, errors='coerce')

# Reemplazar comas por puntos en las columnas numéricas para garantizar correcta interpretación
columns_to_clean = ['Valor', 'Precio ofertado', 'MPO', 'Ajuste', 'Precio ajustado']
for column in columns_to_clean:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column].replace({',': '.'}, regex=True), errors='coerce'
        )

# Eliminar filas con valores nulos (producidos por errores o valores no numéricos)
data = data.dropna(subset=columns_to_clean)

# Filtrar los registros donde ninguna de las columnas seleccionadas sea igual a 0
filtered_data = data[
    (data['MPO'] != 0) &
    (data['Ajuste'] != 0) &
    (data['Precio ajustado'] != 0) &
    (data['Valor'] != 0)
]

# Organizar los datos por fecha y luego por hora
if 'Hora' in filtered_data.columns:
    filtered_data = filtered_data.sort_values(by=['Fecha', 'Hora'])

# Convertir el formato de la columna 'Fecha' a 'dd/mm/yyyy'
filtered_data['Fecha'] = pd.to_datetime(filtered_data['Fecha']).dt.strftime('%d/%m/%Y')

# Verificar el total de registros esperados para febrero de 2016
expected_records = 29 * 24  # Febrero de 2016 tiene 29 días (año bisiesto) y 24 horas por día
if len(filtered_data) == expected_records:
    print(f"El conjunto de datos tiene {len(filtered_data)} registros, como se esperaba.")
else:
    print(f"El conjunto de datos tiene {len(filtered_data)} registros, pero se esperaban {expected_records}.")

# Guardar el resultado en un nuevo archivo CSV
output_path = Path("data/Datasetcalculado2016_filtered.csv")
filtered_data.to_csv(output_path, index=False, sep=';')
print(f"Archivo filtrado guardado en: {output_path}")
