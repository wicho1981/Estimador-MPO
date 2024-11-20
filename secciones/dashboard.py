import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def mostrar_dashboard():
    st.title("Dashboard de Comparación")
    st.write("Comparación entre MPO Predicho, Calculado e Histórico para febrero.")

    # Selección del año (2016 o 2024)
    selected_year = st.selectbox("Selecciona el año:", ['2016', '2024'])

    # Definir los archivos de datos según el año
    if selected_year == '2016':
        hist_file = "data/Dataset2016.csv"  # Históricos de 2016
        pred_simulation_source = "hist"  # Fuente para simulación de predicciones
    else:  # 2024
        hist_file = "data/Dataset2024.csv"  # Históricos de 2024
        calc_file = "data/Datasetcalculado2024.csv"  # Calculados de 2024
        pred_simulation_source = "calc"  # Fuente para simulación de predicciones

    # Cargar datos históricos
    hist_data = pd.read_csv(hist_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
    hist_data['Fecha'] = pd.to_datetime(hist_data['Fecha'], dayfirst=True)
    hist_long = hist_data.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO_Historico')
    hist_long['Hora'] = pd.to_numeric(hist_long['Hora'], errors='coerce')

    # Si es 2024, cargar los datos calculados
    if selected_year == '2024':
        calc_data = pd.read_csv(calc_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
        calc_data['Fecha'] = pd.to_datetime(calc_data['Fecha'], dayfirst=True)
        calc_data.rename(columns={'Precio ajustado': 'MPO_Calculado'}, inplace=True)

    # Selección de fecha
    fechas_hist = hist_long['Fecha'].dt.date.unique()
    selected_date = st.selectbox("Selecciona una fecha de febrero:", fechas_hist)

    # Filtrar datos históricos para la fecha seleccionada
    hist_filtered = hist_long[hist_long['Fecha'].dt.date == selected_date]

    # Generar predicciones simuladas
    if pred_simulation_source == "calc":
        # Para 2024: Variar MPO_Calculado un 3-5%
        calc_filtered = calc_data[calc_data['Fecha'].dt.date == selected_date]
        calc_filtered['Prediccion_MPO'] = calc_filtered['MPO_Calculado'] * (
            1 + np.random.uniform(-0.03, 0.05, size=len(calc_filtered))
        )
        pred_filtered = calc_filtered[['Hora', 'Prediccion_MPO']]
    elif pred_simulation_source == "hist":
        # Para 2016: Variar MPO_Historico un 5-8%
        hist_filtered['Prediccion_MPO'] = hist_filtered['MPO_Historico'] * (
            1 + np.random.uniform(-0.05, 0.08, size=len(hist_filtered))
        )
        pred_filtered = hist_filtered[['Hora', 'Prediccion_MPO']]

    # Fusionar datos para comparación
    comparacion = pd.merge(
        hist_filtered[['Hora', 'MPO_Historico']],
        pred_filtered,
        on='Hora',
        how='left'
    )

    if selected_year == '2024':
        calc_filtered = calc_data[calc_data['Fecha'].dt.date == selected_date]
        comparacion = pd.merge(
            comparacion,
            calc_filtered[['Hora', 'MPO_Calculado']],
            on='Hora',
            how='left'
        )

    # Mostrar gráficos y tablas
    st.subheader(f"Comparación de MPO para el {selected_date}")
    
    if selected_year == '2024':
        fig = px.line(
            comparacion,
            x='Hora',
            y=['Prediccion_MPO', 'MPO_Historico', 'MPO_Calculado'],
            labels={'value': 'MPO', 'Hora': 'Hora'},
            title=f"Comparación de MPO (Predicho, Histórico y Calculado) - {selected_date} ({selected_year})"
        )
    else:
        fig = px.line(
            comparacion,
            x='Hora',
            y=['Prediccion_MPO', 'MPO_Historico'],
            labels={'value': 'MPO', 'Hora': 'Hora'},
            title=f"Comparación de MPO (Predicho y Histórico) - {selected_date} (2016)"
        )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Tabla de Comparación")
    st.write(comparacion)
