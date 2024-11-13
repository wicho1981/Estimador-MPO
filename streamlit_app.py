import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(
    page_title='Visor de Datos MPO',
    page_icon=':electric_plug:',
    layout='wide'
)

# -------------------------------------------------------------------------
# CSS personalizado para ajustar la cabecera
st.markdown(
    """
    <style>
        .cabecera {
            background-color: var(--background-color);
            padding: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 100px;
        }
        .logo {
            flex: 0 0 20%;
            height: 100%;
            max-height: 100px;
            object-fit: contain;
        }
        .titulo {
            flex: 0 0 70%;
            font-size: 1.7em;
            font-weight: bold;
            text-align: right;
            color: var(--text-color);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Cabecera utilizando columnas de Streamlit
with st.container():
    st.markdown('<div class="cabecera">', unsafe_allow_html=True)

    # Ruta para el logo desde la carpeta principal
    logo_path = Path(__file__).parent / 'logo.svg'
    
    if logo_path.is_file():
        # Mostrar el logo si se encuentra
        st.markdown(f'<img src="data:image/svg+xml;base64,{logo_path.read_bytes().decode()}" class="logo">', unsafe_allow_html=True)
    else:
        # Mostrar mensaje si el logo no se encuentra
        st.markdown('<div style="color:red;">Error: Logo no encontrado.</div>', unsafe_allow_html=True)

    # Título del proyecto y autores
    st.markdown(
        """
        <div class="titulo">Estimación del MPO usando Datos de Generación <br> <i>Autores: Cristian Noguera & Jaider Sanchez</i></div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Añadir un divisor horizontal
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
# Hasta aquí, el código solicitado sin las secciones posteriores.

# -------------------------------------------------------------------------
# Sección de Históricos
if selected_section == 'Históricos':
    st.title(":electric_plug: MPO Históricos")
    
    st.write("""
    Esta sección muestra los datos históricos de MPO en una gráfica de líneas para un día seleccionado y permite filtrar un rango de días para análisis tabular.
    """)

    # Paso 1: Seleccionar el año (2016 o 2024)
    selected_year = st.selectbox('Selecciona el año:', ['2016', '2024'])

    # Cargar los datos según el año seleccionado
    if selected_year == '2016':
        mpo_data_df = mpo_data_2016
    else:
        mpo_data_df = mpo_data_2024

    # Paso 2: Gráfico - Seleccionar la fecha para una gráfica de un solo día
    st.subheader("Gráfica de líneas para el MPO durante un día específico")

    unique_dates = mpo_data_df['Fecha'].dt.date.unique()  # Asegurar que solo se muestren fechas
    selected_date = st.selectbox('Selecciona la fecha para la gráfica:', unique_dates)

    # Filtrar los datos para el día seleccionado
    filtered_mpo_data = mpo_data_df[mpo_data_df['Fecha'].dt.date == selected_date]

    # Convertir los datos de formato ancho a largo para graficar cada hora como un punto separado
    long_data = pd.melt(filtered_mpo_data, id_vars=['Fecha'], var_name='Hora', value_name='MPO')

    # Convertir la columna 'Hora' a numérico y eliminar datos no válidos
    long_data['Hora'] = pd.to_numeric(long_data['Hora'], errors='coerce')
    long_data['MPO'] = pd.to_numeric(long_data['MPO'], errors='coerce')

    # Eliminar filas con datos faltantes o no válidos
    long_data.dropna(subset=['MPO'], inplace=True)

    # Comprobar si hay datos para graficar
    if long_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        # Crear el gráfico de líneas usando 'Hora' como eje X y 'MPO' como eje Y
        line_chart = px.line(long_data, x='Hora', y='MPO', title=f'MPO durante el {selected_date.strftime("%d/%m/%Y")}', height=600)

        # Mostrar la gráfica
        st.plotly_chart(line_chart, use_container_width=True)

# -------------------------------------------------------------------------
# Sección de Calculados
elif selected_section == 'Calculados':
    st.title("Datos Calculados")
    st.write("Esta sección muestra los datos calculados de MPO ajustado para un día específico.")

    # Paso 1: Gráfico - Seleccionar la fecha para una gráfica de un solo día
    st.subheader("Gráfica de líneas para el Precio Ajustado durante un día específico")

    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()  # Asegurarse de mostrar solo fechas únicas
    selected_date_calculado = st.selectbox('Selecciona la fecha para la gráfica (calculados):', unique_dates_calculado)

    # Filtrar los datos para el día seleccionado
    filtered_mpo_data_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_calculado]

    # Convertir los datos a formato largo para graficar cada hora como un punto separado
    long_data_calculado = filtered_mpo_data_calculado[['Hora', 'Precio ajustado']].copy()

    # Convertir la columna 'Hora' a numérico y eliminar datos no válidos
    long_data_calculado['Hora'] = pd.to_numeric(long_data_calculado['Hora'], errors='coerce')
    long_data_calculado['Precio ajustado'] = pd.to_numeric(long_data_calculado['Precio ajustado'], errors='coerce')

    # Comprobar si hay datos para graficar
    if long_data_calculado.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        # Crear el gráfico de líneas usando 'Hora' como eje X y 'Precio ajustado' como eje Y
        line_chart_calculado = px.line(long_data_calculado, x='Hora', y='Precio ajustado',
                                       title=f'Precio Ajustado durante el {selected_date_calculado.strftime("%d/%m/%Y")}', 
                                       height=600)

        # Mostrar la gráfica
        st.plotly_chart(line_chart_calculado, use_container_width=True)

# -------------------------------------------------------------------------
# Sección de Predecidos
# Sección de Predecidos (próximamente)
elif selected_section == 'Predecidos':
    # Título y descripción de la sección
    st.title("Predicción del MPO para un Día Completo")

    # Selección de la fecha para predicción
    fecha_seleccionada = st.date_input("Selecciona una fecha para predecir el MPO")

    # Formatear la fecha seleccionada en formato 'dd/mm/aaaa' para mostrarla
    fecha_seleccionada_str = fecha_seleccionada.strftime('%d/%m/%Y')

    # Botón para realizar la predicción
    if st.button("Predecir MPO para todas las horas"):
        # Realizar las predicciones para la fecha seleccionada
        predicciones_dia = predecir_mpo_futuro(fecha_seleccionada_str)

        if predicciones_dia is None:
            st.write(f"No hay datos disponibles para la fecha {fecha_seleccionada_str}.")
        else:
            # Mostrar la tabla de resultados en Streamlit
            st.subheader(f"Predicción de MPO para el día {fecha_seleccionada_str}")
            st.write(predicciones_dia)

            # Gráfico de los valores predichos
            st.subheader("Gráfico de MPO Predicho por Hora")
            st.line_chart(predicciones_dia.set_index('Hora')['MPO Predicho'])

# -------------------------------------------------------------------------
# Sección de Dashboard (Comparación de MPO Histórico y Calculado)
elif selected_section == 'Dashboard':
    st.title("Dashboard: Comparación de MPO Histórico y Calculado")
    st.write("Esta sección compara el MPO histórico y el MPO calculado para un día seleccionado del 2024.")

    # Fechas disponibles en los dos datasets (históricos y calculados)
    unique_dates_2024 = mpo_data_2024['Fecha'].dt.date.unique()
    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()

    # Encontrar las fechas comunes en ambos conjuntos de datos
    common_dates = list(set(unique_dates_2024).intersection(set(unique_dates_calculado)))
    common_dates.sort()  # Ordenar las fechas comunes

    # Selección de la fecha para la comparación
    selected_date_dashboard = st.selectbox('Selecciona la fecha para la comparación:', common_dates)

    # Filtrar los datos para la fecha seleccionada
    filtered_mpo_historico = mpo_data_2024[mpo_data_2024['Fecha'].dt.date == selected_date_dashboard]
    filtered_mpo_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_dashboard]

    # Transformar el dataset histórico (columnas de horas 0-23) a formato largo
    long_data_historico = filtered_mpo_historico.melt(
        id_vars=['Fecha'], value_vars=[str(i) for i in range(24)], var_name='Hora', value_name='MPO'
    )

    # Asegurar que la columna 'Hora' es numérica
    long_data_historico['Hora'] = pd.to_numeric(long_data_historico['Hora'])

    # Preparar los datos calculados
    long_data_calculado = filtered_mpo_calculado[['Hora', 'Precio ajustado']].copy()
    long_data_calculado.rename(columns={'Precio ajustado': 'MPO'}, inplace=True)
    long_data_calculado['Fuente'] = 'Calculado'

    # Añadir fuente a los datos históricos
    long_data_historico['Fuente'] = 'Histórico'

    # Concatenar ambos datasets para tener una tabla lista para graficar
    combined_data = pd.concat([long_data_historico, long_data_calculado], ignore_index=True)

    # Comprobar si hay datos para graficar
    if combined_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        # Crear el gráfico de líneas comparando MPO Histórico y Calculado
        comparison_chart = px.line(combined_data, x='Hora', y='MPO', color='Fuente',
                                   title=f'Comparación del MPO Histórico y Calculado durante el {selected_date_dashboard.strftime("%d/%m/%Y")}',
                                   height=600)

        # Mostrar la gráfica
        st.plotly_chart(comparison_chart, use_container_width=True)

        # Calcular las diferencias entre MPO Histórico y Calculado
        st.subheader("Tabla de diferencias entre MPO Histórico y Calculado")
        merged_data = pd.merge(long_data_historico[['Hora', 'MPO']], long_data_calculado[['Hora', 'MPO']],
                               on='Hora', suffixes=('_Historico', '_Calculado'))

        merged_data['Diferencia'] = merged_data['MPO_Historico'] - merged_data['MPO_Calculado']
        st.dataframe(merged_data, use_container_width=True)

        # Botón para descargar la tabla de diferencias
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
