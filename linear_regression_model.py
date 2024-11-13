# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Función para entrenar y predecir con el modelo de regresión lineal
def train_linear_regression_model(data):
    # Seleccionar las columnas necesarias para el modelo
    X = data[['Valor', 'Hora', 'Tecnologia', 'Ajuste']]  # Reemplazar con las columnas que consideres necesarias
    y = data['MPO']

    # Convertir las variables categóricas en variables dummy (en este caso, la columna 'Tecnologia')
    X = pd.get_dummies(X, columns=['Tecnologia'], drop_first=True)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_test, y_pred, mse, r2
