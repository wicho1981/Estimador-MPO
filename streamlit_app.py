import streamlit as st
import pandas as pd
from secciones.cabecera import mostrar_cabecera
from secciones.dashboard import mostrar_dashboard
from secciones.historicos import mostrar_historicos
from secciones.calculados import mostrar_calculados
from secciones.predecidos import mostrar_predecidos

# Page configuration
st.set_page_config(page_title='MPO Data Viewer', page_icon=':electric_plug:', layout='wide')

# Load data
mpo_data_2016 = pd.read_csv("data/Dataset2016.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_2016['Fecha'] = pd.to_datetime(mpo_data_2016['Fecha'], dayfirst=True)

mpo_data_2024 = pd.read_csv("data/Dataset2024.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_2024['Fecha'] = pd.to_datetime(mpo_data_2024['Fecha'], dayfirst=True)

mpo_data_calculado2024 = pd.read_csv("data/Datasetcalculado2024.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_calculado2024['Fecha'] = pd.to_datetime(mpo_data_calculado2024['Fecha'], dayfirst=True)

mpo_data_calculado2016 = pd.read_csv("data/Datasetcalculado2016.csv", delimiter=';', decimal='.', encoding='ISO-8859-1')
mpo_data_calculado2016['Fecha'] = pd.to_datetime(mpo_data_calculado2016['Fecha'], dayfirst=True)

# Show header
mostrar_cabecera()

# Section selection menu
selected_section = st.selectbox('Select the section:', ['Dashboard', 'Historical Data', 'Calculated Data', 'Predicted Data'])

# Display the selected section
if selected_section == 'Dashboard':
    mostrar_dashboard()
elif selected_section == 'Historical Data':
    mostrar_historicos(mpo_data_2016, mpo_data_2024)
elif selected_section == 'Calculated Data':
    mostrar_calculados(mpo_data_calculado2024,mpo_data_calculado2016)
elif selected_section == 'Predicted Data':
    mostrar_predecidos()