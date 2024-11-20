import streamlit as st
import pandas as pd
from data.entrenar_y_predecir import cargar_datos
from secciones.cabecera import mostrar_cabecera
from secciones.dashboard import mostrar_dashboard
from secciones.historicos import mostrar_historicos
from secciones.calculados import mostrar_calculados
from secciones.predecidos import mostrar_predecidos

# Configuración de la página
st.set_page_config(page_title='Visor de Datos MPO', page_icon=':electric_plug:', layout='wide')

# Cargar datos
mpo_data_2016 = pd.read_csv("data/Dataset2016.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_2016['Fecha'] = pd.to_datetime(mpo_data_2016['Fecha'], dayfirst=True)

mpo_data_2024 = pd.read_csv("data/Dataset2024.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_2024['Fecha'] = pd.to_datetime(mpo_data_2024['Fecha'], dayfirst=True)

mpo_data_calculado = pd.read_csv("data/Datasetcalculado2024.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_calculado['Fecha'] = pd.to_datetime(mpo_data_calculado['Fecha'], dayfirst=True)

# Mostrar cabecera
mostrar_cabecera()

# Menú de selección de sección
selected_section = st.selectbox('Selecciona la sección:', ['Dashboard', 'Históricos', 'Calculados', 'Predecidos'])

# Mostrar la sección seleccionada
if selected_section == 'Dashboard':
    mostrar_dashboard()
elif selected_section == 'Históricos':
    mostrar_historicos(mpo_data_2016, mpo_data_2024)
elif selected_section == 'Calculados':
    mostrar_calculados(mpo_data_calculado)
elif selected_section == 'Predecidos':
    mostrar_predecidos()
