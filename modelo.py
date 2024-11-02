import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------
# Paso 1: Cargar los datos
# ----------------------------

# Cargar los CSVs como dataframes desde la carpeta 'data' con encoding='ISO-8859-1' y delimitador ';'
aportes_2016 = pd.read_csv('data/Aportes2016.csv', encoding='ISO-8859-1', delimiter=';')
aportes_2024 = pd.read_csv('data/Aportes2024.csv', encoding='ISO-8859-1', delimiter=';')
dataset_2016 = pd.read_csv('data/dataset2016.csv', encoding='ISO-8859-1', delimiter=';')
dataset_2024 = pd.read_csv('data/Dataset2024.csv', encoding='ISO-8859-1', delimiter=';')
dataset_calculado_2024 = pd.read_csv('data/Datasetcalculado2024.csv', encoding='ISO-8859-1', delimiter=';')
listado_generacion = pd.read_csv('data/ListadoGeneracion.csv', encoding='ISO-8859-1', delimiter=';')
mpo_data = pd.read_csv('data/mpo_data.csv', encoding='ISO-8859-1', delimiter=';')
pre_oferta_2016 = pd.read_csv('data/preofe2016.csv', encoding='ISO-8859-1', delimiter=';')
pre_oferta_2024 = pd.read_csv('data/preofe2024.csv', encoding='ISO-8859-1', delimiter=';')
reservas_2016 = pd.read_csv('data/Reservas2016.csv', encoding='ISO-8859-1', delimiter=';')
reservas_2024 = pd.read_csv('data/Reservas2024.csv', encoding='ISO-8859-1', delimiter=';')

# ----------------------------
# Paso 2: Integrar los datos
# ----------------------------

# Usamos Datasetcalculado2024.csv como base para integrar las demás variables
# Agregamos columnas de otros CSVs basándonos en 'Fecha', 'Código Agente' y 'Recurso' donde sea posible

# Unión con precios de oferta del 2024
data = pd.merge(dataset_calculado_2024, pre_oferta_2024, on=['Fecha', 'Código Agente', 'Recurso'], how='left')

# Unión con aportes hídricos del 2024
data = pd.merge(data, aportes_2024[['Fecha', 'Aportes Energía kWh', 'Aportes Caudal m3/s']], on='Fecha', how='left')

# Unión con reservas de embalses del 2024
data = pd.merge(data, reservas_2024[['Fecha', 'Volumen Útil Diario Energía kWh', 'Volumen Útil Diario %']], on='Fecha', how='left')

# Unión con listado de generación para obtener capacidad y tipo de generación
data = pd.merge(data, listado_generacion[['Recurso', 'Capacidad Efectiva Neta [MW]', 'Tipo Generación']], on='Recurso', how='left')

# Verificar las columnas resultantes
print("Columnas después de integrar los datos:", data.columns)

# ----------------------------
# Paso 3: Selección de variables (features y target)
# ----------------------------

# Selección de variables de entrada y la variable objetivo
X = data[['Precio ofertado', 'Aportes Energía kWh', 'Volumen Útil Diario Energía kWh', 
          'Capacidad Efectiva Neta [MW]', 'Aportes Caudal m3/s', 'Volumen Útil Diario %']]
y = data['MPO']  # Variable objetivo

# Eliminar filas con valores faltantes para asegurar un entrenamiento sin errores
X = X.dropna()
y = y[X.index]

# ----------------------------
# Paso 4: Normalización y división de datos
# ----------------------------

# Normalización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# Paso 5: Entrenamiento del modelo de regresión
# ----------------------------

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción en los datos de prueba
y_pred = model.predict(X_test)

# ----------------------------
# Paso 6: Evaluación del modelo
# ----------------------------

# Calcular el error cuadrático medio y el error absoluto medio
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Mostrar los coeficientes del modelo para interpretar la importancia de cada variable
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print("\nCoeficientes del modelo de regresión lineal:")
print(coef_df)

# ----------------------------
# Paso 7: Visualización de Resultados (Opcional)
# ----------------------------

# Comparar las predicciones con los valores reales (solo para primeras filas)
comparison_df = pd.DataFrame({'Valor Real': y_test, 'Predicción': y_pred})
print("\nComparación entre valores reales y predicciones:")
print(comparison_df.head(10))
