import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Entrenar el modelo solo una vez y guardarlo para futuras predicciones
def entrenar_modelo():
    # Cargar y preparar los datos
    dataset_calculado_2024 = pd.read_csv('data/Datasetcalculado2024.csv', delimiter=';', encoding='ISO-8859-1')
    
    # Selección de variables (features y target)
    X = dataset_calculado_2024[['Precio ofertado', 'Valor', 'Ajuste', 'Precio ajustado']]
    y = dataset_calculado_2024['MPO']

    # Eliminar filas con valores faltantes
    X = X.dropna()
    y = y[X.index]

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Guardar el modelo y el scaler para usarlos en el futuro
    joblib.dump(model, 'modelo_mpo.pkl')
    joblib.dump(scaler, 'scaler_mpo.pkl')

# Función para predecir el MPO para cada hora en un día futuro usando los datos históricos como referencia
def predecir_mpo_futuro(fecha):
    # Cargar el modelo y el scaler
    model = joblib.load('modelo_mpo.pkl')
    scaler = joblib.load('scaler_mpo.pkl')

    # Para predecir una fecha futura, generaremos las entradas correspondientes para ese día
    # Aquí asumimos que las características históricas se mantendrán relativamente similares.
    # Por ejemplo, si tenemos características promedio o valores fijos, podemos usarlos para predicciones futuras.
    # A continuación generaremos un dataframe con 24 entradas (una por cada hora del día).

    # Datos ficticios para las características del modelo
    # Puedes ajustar estos valores basándote en la media o la mediana de los datos históricos.
    # Aquí utilizo valores promedio para generar predicciones.
    promedio_precio_ofertado = 690.1761599  # Valor promedio de "Precio ofertado" (puedes cambiarlo por el valor real calculado)
    promedio_valor = 500000  # Valor promedio de "Valor" (puedes ajustarlo)
    promedio_ajuste = 615.2351599  # Valor promedio de "Ajuste" (puedes cambiarlo por el valor real calculado)
    promedio_precio_ajustado = 615.2351599  # Valor promedio de "Precio ajustado"

    # Crear un DataFrame con 24 filas (una para cada hora del día)
    horas = list(range(24))
    datos_futuros = pd.DataFrame({
        'Hora': horas,
        'Precio ofertado': [promedio_precio_ofertado] * 24,
        'Valor': [promedio_valor] * 24,
        'Ajuste': [promedio_ajuste] * 24,
        'Precio ajustado': [promedio_precio_ajustado] * 24
    })

    # Seleccionar las columnas necesarias para las predicciones
    X_futuro = datos_futuros[['Precio ofertado', 'Valor', 'Ajuste', 'Precio ajustado']]

    # Normalizar los datos
    X_futuro_scaled = scaler.transform(X_futuro)

    # Realizar las predicciones para cada hora del día futuro
    predicciones = model.predict(X_futuro_scaled)

    # Añadir las predicciones al DataFrame original
    datos_futuros['MPO Predicho'] = predicciones
    datos_futuros['Fecha'] = fecha  # Añadir la fecha seleccionada

    return datos_futuros[['Fecha', 'Hora', 'MPO Predicho']]  # Retorna solo las horas y los MPO predichos

# Entrena el modelo solo una vez (comenta esta línea después de entrenar)
entrenar_modelo()
