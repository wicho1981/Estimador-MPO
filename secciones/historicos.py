import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_historicos(mpo_data_2016, mpo_data_2024):
    st.title("Históricos")
    st.write("Visualización de datos históricos de MPO.")
    
    # Selección de año
    selected_year = st.selectbox('Selecciona el año:', ['2016', '2024'])
    mpo_data_df = mpo_data_2016 if selected_year == '2016' else mpo_data_2024

    # Selección de fecha para el año seleccionado
    unique_dates = mpo_data_df['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Selecciona una fecha:', unique_dates)
    
    # Filtrar y graficar datos de la fecha seleccionada
    filtered_data = mpo_data_df[mpo_data_df['Fecha'].dt.date == selected_date]
    long_data = filtered_data.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO')
    long_data['Hora'] = pd.to_numeric(long_data['Hora'], errors='coerce')

    fig = px.line(long_data, x='Hora', y='MPO', title=f"Datos históricos de MPO - {selected_date} ({selected_year})")
    st.plotly_chart(fig, use_container_width=True)
