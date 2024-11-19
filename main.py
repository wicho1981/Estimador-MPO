from data.entrenar_y_predecir import cargar_datos, entrenar_modelo, generar_predicciones

# Cargar los datos
X, y, scaler_X, scaler_y = cargar_datos()

# Entrenar el modelo
n_inputs = X.shape[1]
anfis_model = entrenar_modelo(X, y, n_inputs)

# Generar predicciones para años específicos
for year in [2016, 2024, 2025]:
    generar_predicciones(anfis_model, scaler_X, scaler_y, year)
