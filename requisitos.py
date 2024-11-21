import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


def verificar_relacion_lineal(dataset, features, target):
    """Verifica visualmente la relación lineal entre variables predictoras y objetivo."""
    print("\n1. Verificación de Relación Lineal")
    sns.pairplot(dataset, x_vars=features, y_vars=target, height=5, aspect=0.7)
    plt.suptitle('Relación Lineal entre Predictores y Objetivo', y=1.02)
    plt.show()


def verificar_independencia(model):
    """Verifica la independencia de los errores usando el test de Durbin-Watson."""
    print("\n2. Verificación de Independencia de los Errores (Durbin-Watson Test)")
    dw_stat = durbin_watson(model.resid)
    print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
    if dw_stat < 1.5 or dw_stat > 2.5:
        print("- Alerta: Existe posible autocorrelación entre los residuos.")
    else:
        print("- No se detecta una fuerte autocorrelación.")
    return dw_stat


def verificar_homoscedasticidad(model):
    """Verifica visualmente la homoscedasticidad mediante un gráfico de residuos."""
    print("\n3. Verificación de Homoscedasticidad (Gráfico de Residuos)")
    plt.scatter(model.fittedvalues, model.resid, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos vs Valores Predichos')
    plt.show()


def verificar_normalidad_residuos(model):
    """Verifica la normalidad de los errores mediante Q-Q Plot y Shapiro-Wilk Test."""
    print("\n4. Verificación de Normalidad de los Residuos")
    stats.probplot(model.resid, dist="norm", plot=plt)
    plt.title('Gráfico Q-Q de los Residuos')
    plt.show()
    shapiro_test = stats.shapiro(model.resid)
    print(f"Shapiro-Wilk Test: W={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("- Alerta: Los residuos no siguen una distribución normal.")
    else:
        print("- Los residuos parecen seguir una distribución normal.")
    return shapiro_test.pvalue


def calcular_vif(X):
    """Calcula el Variance Inflation Factor (VIF) para las variables predictoras."""
    print("\n5. Verificación de Colinealidad (Variance Inflation Factor - VIF)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    print(vif_data)
    if any(vif_data['VIF'] > 10):
        print("- Alerta: Se detecta colinealidad alta en algunas variables independientes.")
    else:
        print("- No se detecta colinealidad significativa entre las variables independientes.")
    return vif_data


def check_regression_assumptions(dataset_path):
    # Cargar el dataset
    dataset = pd.read_csv(dataset_path, delimiter=';', encoding='ISO-8859-1')
    
    # Seleccionar las variables predictoras (features) y la variable objetivo (target)
    features = ['Precio ofertado', 'Valor', 'Ajuste', 'Precio ajustado']
    target = 'MPO'
    X = dataset[features]
    y = dataset[target]
    
    # 1. Relación Lineal
    verificar_relacion_lineal(dataset, features, target)
    
    # 2. Ajustar el modelo de regresión
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    # 3. Verificaciones del modelo
    dw_stat = verificar_independencia(model)
    verificar_homoscedasticidad(model)
    p_value_normalidad = verificar_normalidad_residuos(model)
    vif_data = calcular_vif(X)

    # Resumen Final
    print("\nResumen de la viabilidad para Regresión Lineal:")
    if (1.5 <= dw_stat <= 2.5) and (p_value_normalidad >= 0.05) and all(vif_data['VIF'] <= 10):
        print("El dataset parece cumplir con los supuestos de regresión lineal. Es viable proceder.")
    else:
        print("El dataset no cumple completamente con los supuestos. Se recomienda revisar los puntos indicados antes de proceder.")

# Llama a la función pasando el camino del archivo CSV
check_regression_assumptions('data/Datasetcalculado2024.csv')
