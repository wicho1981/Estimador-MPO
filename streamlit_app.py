import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuración de la página
st.set_page_config(
    page_title='Visor de Datos MPO',
    page_icon=':electric_plug:',
    layout='wide'
)

# Añadir un divisor superior
st.markdown("---") 

# Crear tres columnas para la cabecera
col1, col2, col3 = st.columns([1, 0.1, 3])

# Columna 1: Logo con ajuste de tamaño
with col1:
    logo_path = Path(__file__).parent / 'logo.png'
    if logo_path.is_file():
        st.image(str(logo_path), use_column_width=True)  # Logo ocupa toda la anchura de la columna
    else:
        st.write("Error: Logo no encontrado")

# Columna 2: Línea vertical (usando markdown para crear una línea estrecha)
with col2:
    st.markdown("<div style='height: 100%; width: 2px; background-color: #cccccc;'></div>", unsafe_allow_html=True)

# Columna 3: Título y subtítulo alineados a la derecha y con tamaño ajustado
with col3:
    st.markdown(
        """
        <div style='text-align: right;'>
            <h2 style='margin-bottom: 5px; color: #333333; font-size: 1.6em;'>Estimación del Máximo Precio Ofertado en el Mercado de Energía Mayorista Colombiano apoyado en IA</h2>
            <p style='margin-top: 0px; color: #555555; font-size: 1.2em; font-style: italic;'>Autores: Cristian Noguera & Jaider Sanchez</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Añadir un divisor inferior
st.markdown("---")

# -------------------------------------------------------------------------
# Función para cargar los datos CSV
@st.cache_data
def get_mpo_data(file_name):
    DATA_FILENAME = Path(__file__).parent / f'data/{file_name}'
    raw_data = pd.read_csv(DATA_FILENAME, delimiter=';', decimal='.', encoding='ISO-8859-1')

    if 'Fecha' in raw_data.columns:
        raw_data['Fecha'] = pd.to_datetime(raw_data['Fecha'], dayfirst=True)

    return raw_data

# Cargar los datos de los años 2016 y 2024
mpo_data_2016 = get_mpo_data('dataset2016.csv')
mpo_data_2024 = get_mpo_data('Dataset2024.csv')
mpo_data_calculado = get_mpo_data('Datasetcalculado2024.csv')

# -------------------------------------------------------------------------
# Menú desplegable vertical utilizando 'selectbox'
selected_section = st.selectbox('Selecciona la sección:', ['Dashboard', 'Históricos', 'Calculados', 'Predecidos'])

# -------------------------------------------------------------------------
# Sección de Históricos
if selected_section == 'Históricos':
    st.title(":electric_plug: MPO Históricos")
    
    st.write("""
    Esta sección muestra los datos históricos de MPO en una gráfica de líneas para un día seleccionado y permite filtrar un rango de días para análisis tabular.
    """)

    selected_year = st.selectbox('Selecciona el año:', ['2016', '2024'])

    mpo_data_df = mpo_data_2016 if selected_year == '2016' else mpo_data_2024

    st.subheader("Gráfica de líneas para el MPO durante un día específico")

    unique_dates = mpo_data_df['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Selecciona la fecha para la gráfica:', unique_dates)

    filtered_mpo_data = mpo_data_df[mpo_data_df['Fecha'].dt.date == selected_date]

    long_data = pd.melt(filtered_mpo_data, id_vars=['Fecha'], var_name='Hora', value_name='MPO')

    long_data['Hora'] = pd.to_numeric(long_data['Hora'], errors='coerce')
    long_data['MPO'] = pd.to_numeric(long_data['MPO'], errors='coerce')

    long_data.dropna(subset=['MPO'], inplace=True)

    if long_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        line_chart = px.line(long_data, x='Hora', y='MPO', title=f"MPO durante el {selected_date.strftime('%d/%m/%Y')}", height=600)
        st.plotly_chart(line_chart, use_container_width=True)

# -------------------------------------------------------------------------
# Sección de Calculados
elif selected_section == 'Calculados':
    st.title("Datos Calculados")
    st.write("Esta sección muestra los datos calculados de MPO ajustado para un día específico.")

    st.subheader("Gráfica de líneas para el Precio Ajustado durante un día específico")

    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()
    selected_date_calculado = st.selectbox('Selecciona la fecha para la gráfica (calculados):', unique_dates_calculado)

    filtered_mpo_data_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_calculado]

    long_data_calculado = filtered_mpo_data_calculado[['Hora', 'Precio ajustado']].copy()

    long_data_calculado['Hora'] = pd.to_numeric(long_data_calculado['Hora'], errors='coerce')
    long_data_calculado['Precio ajustado'] = pd.to_numeric(long_data_calculado['Precio ajustado'], errors='coerce')

    if long_data_calculado.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        line_chart_calculado = px.line(long_data_calculado, x='Hora', y='Precio ajustado',
                                       title=f"Precio Ajustado durante el {selected_date_calculado.strftime('%d/%m/%Y')}", 
                                       height=600)
        st.plotly_chart(line_chart_calculado, use_container_width=True)

# -------------------------------------------------------------------------
# Sección de Predecidos con opción de Test para generar datos aleatorios
if selected_section == 'Predecidos':
    st.title("Predicción del MPO para un Día Completo")

    # Selección de la fecha para predicción
    fecha_seleccionada = st.date_input("Selecciona una fecha para predecir el MPO")
    fecha_seleccionada_str = fecha_seleccionada.strftime('%d/%m/%Y')

    # Botón para activar la simulación con datos aleatorios
    if st.button("Generar Predicción (Simulación de Test)"):
        st.write(f"Simulación de predicción para el día {fecha_seleccionada_str} basados en datos históricos.")

        # Función para generar valores aleatorios de predicción (simulación)
        def generate_random_predictions(fecha_seleccionada, data):
            # Tomar valores históricos de MPO para generar aleatoriamente
            mpo_values = data['MPO'].dropna().values
            random_predictions = []

            for hora in range(24):
                mpo_random = np.random.choice(mpo_values)  # Selecciona aleatoriamente un MPO histórico
                error_random = np.random.uniform(-5, 5)    # Agrega un error aleatorio entre -5 y 5
                prediccion = mpo_random + error_random
                random_predictions.append({'Hora': hora, 'MPO_Prediccion': prediccion})

            return pd.DataFrame(random_predictions)

        # Generar predicciones aleatorias
        predicciones_df = generate_random_predictions(fecha_seleccionada, mpo_data_calculado)
        
        # Mostrar gráfico y tabla con la simulación
        fig = px.line(predicciones_df, x='Hora', y='MPO_Prediccion', title=f"Simulación de predicción de MPO para el {fecha_seleccionada_str}", labels={'MPO_Prediccion': 'MPO'})
        st.plotly_chart(fig)
        st.dataframe(predicciones_df)

# -------------------------------------------------------------------------
# Sección de Dashboard (Comparación de MPO Histórico y Calculado)
elif selected_section == 'Dashboard':
    st.title("Dashboard: Comparación de MPO Histórico y Calculado")
    st.write("Esta sección compara el MPO histórico y el MPO calculado para un día seleccionado del 2024.")

    unique_dates_2024 = mpo_data_2024['Fecha'].dt.date.unique()
    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()

    common_dates = list(set(unique_dates_2024).intersection(set(unique_dates_calculado)))
    common_dates.sort()

    selected_date_dashboard = st.selectbox('Selecciona la fecha para la comparación:', common_dates)

    filtered_mpo_historico = mpo_data_2024[mpo_data_2024['Fecha'].dt.date == selected_date_dashboard]
    filtered_mpo_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_dashboard]

    long_data_historico = filtered_mpo_historico.melt(
        id_vars=['Fecha'], value_vars=[str(i) for i in range(24)], var_name='Hora', value_name='MPO'
    )
    long_data_historico['Hora'] = pd.to_numeric(long_data_historico['Hora'])

    long_data_calculado = filtered_mpo_calculado[['Hora', 'Precio ajustado']].copy()
    long_data_calculado.rename(columns={'Precio ajustado': 'MPO'}, inplace=True)
    long_data_calculado['Fuente'] = 'Calculado'

    long_data_historico['Fuente'] = 'Histórico'

    combined_data = pd.concat([long_data_historico, long_data_calculado], ignore_index=True)

    if combined_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        comparison_chart = px.line(combined_data, x='Hora', y='MPO', color='Fuente',
                                   title=f"Comparación del MPO Histórico y Calculado durante el {selected_date_dashboard.strftime('%d/%m/%Y')}",
                                   height=600)
        st.plotly_chart(comparison_chart, use_container_width=True)

        st.subheader("Tabla de diferencias entre MPO Histórico y Calculado")
        merged_data = pd.merge(long_data_historico[['Hora', 'MPO']], long_data_calculado[['Hora', 'MPO']],
                               on='Hora', suffixes=('_Historico', '_Calculado'))

        merged_data['Diferencia'] = merged_data['MPO_Historico'] - merged_data['MPO_Calculado']
        st.dataframe(merged_data, use_container_width=True)

        output_csv_diferencias = merged_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar tabla de diferencias como CSV",
            data=output_csv_diferencias,
            file_name=f"diferencias_mpo_{selected_date_dashboard.strftime('%d_%m_%Y')}.csv",
            mime="text/csv"
        )

        output_excel_diferencias = io.BytesIO()
        with pd.ExcelWriter(output_excel_diferencias, engine='xlsxwriter') as writer:
            merged_data.to_excel(writer, index=False, sheet_name='Diferencias')
            worksheet = writer.sheets['Diferencias']
            for i, col in enumerate(merged_data.columns):
                max_len = max(merged_data[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)

        output_excel_diferencias.seek(0)
        st.download_button(
            label="📥 Descargar tabla de diferencias como Excel",
            data=output_excel_diferencias,
            file_name=f"diferencias_mpo_{selected_date_dashboard.strftime('%d_%m_%Y')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
