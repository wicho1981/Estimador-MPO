import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import io

# Configuración de la página
st.set_page_config(
    page_title='Visor de Datos MPO',
    page_icon=':electric_plug:',
    layout='wide'  # Activar el modo amplio
)

# -----------------------------------------------------------------------------
# Función para cargar los datos CSV
@st.cache_data
def get_mpo_data(file_name):
    """Carga los datos MPO desde el archivo CSV especificado."""
    # Ruta del archivo CSV
    DATA_FILENAME = Path(__file__).parent / f'data/{file_name}'
    # Leer el archivo CSV
    raw_data = pd.read_csv(DATA_FILENAME, delimiter=';', decimal='.', encoding='utf-8')

    # Convertir la columna 'Fecha' al formato datetime
    if 'Fecha' in raw_data.columns:
        raw_data['Fecha'] = pd.to_datetime(raw_data['Fecha'], dayfirst=True)

    return raw_data

# Cargar los datos de los años 2016 y 2024
mpo_data_2016 = get_mpo_data('dataset2016.csv')
mpo_data_2024 = get_mpo_data('Dataset2024.csv')

# -----------------------------------------------------------------------------
# Menú desplegable vertical utilizando 'selectbox'
selected_section = st.selectbox('Selecciona la sección:', ['Históricos', 'Calculados', 'Predecidos'])

# -----------------------------------------------------------------------------
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

    # Formatear la fecha como 'dd/mm/aaaa'
    long_data['Fecha'] = long_data['Fecha'].dt.strftime('%d/%m/%Y')

    # Comprobar si hay datos para graficar
    if long_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otro día.")
    else:
        # Crear el gráfico de líneas usando 'Hora' como eje X y 'MPO' como eje Y
        line_chart = px.line(long_data, x='Hora', y='MPO', title=f'MPO durante el {selected_date.strftime("%d/%m/%Y")}', height=600)

        # Mostrar la gráfica
        st.plotly_chart(line_chart, use_container_width=True)

    # Paso 3: Tabla - Seleccionar un rango de fechas para el análisis tabular
    st.subheader("Vista tabular de los datos MPO para un rango de fechas")

    # Filtrar por rango de fechas
    min_date = mpo_data_df['Fecha'].min().date()
    max_date = mpo_data_df['Fecha'].max().date()

    # Seleccionar rango de fechas para la tabla
    selected_date_range = st.date_input(
        'Selecciona el rango de fechas para la tabla:',
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Convertir las fechas seleccionadas a formato datetime
    selected_start_date = pd.to_datetime(selected_date_range[0])
    selected_end_date = pd.to_datetime(selected_date_range[1])

    # Filtrar los datos basados en el rango de fechas seleccionado
    filtered_table_data = mpo_data_df[
        (mpo_data_df['Fecha'] >= selected_start_date) &
        (mpo_data_df['Fecha'] <= selected_end_date)
    ]

    # Mostrar la tabla filtrada
    if filtered_table_data.empty:
        st.warning("No hay datos disponibles para el rango de fechas seleccionado. Por favor, elige otro rango.")
    else:
        # Formatear las fechas como 'dd/mm/aaaa'
        filtered_table_data['Fecha'] = filtered_table_data['Fecha'].dt.strftime('%d/%m/%Y')
        
        st.write(f"Datos filtrados desde el {selected_start_date.strftime('%d/%m/%Y')} hasta el {selected_end_date.strftime('%d/%m/%Y')}:")
        st.dataframe(filtered_table_data, use_container_width=True)

        # Botón para exportar los datos a Excel
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            filtered_table_data.to_excel(writer, index=False, sheet_name='Datos Filtrados')

            # Ajustar el ancho de las columnas
            worksheet = writer.sheets['Datos Filtrados']
            for i, col in enumerate(filtered_table_data.columns):
                column_len = filtered_table_data[col].astype(str).str.len().max()
                column_len = max(column_len, len(col)) + 2
                worksheet.set_column(i, i, column_len)
        
        output_excel.seek(0)
        
        st.download_button(
            label="📥 Descargar datos filtrados como Excel",
            data=output_excel,
            file_name=f"datos_filtrados_MPO_{selected_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Botón para exportar los datos a CSV
        output_csv = filtered_table_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos filtrados como CSV",
            data=output_csv,
            file_name=f"datos_filtrados_MPO_{selected_year}.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# Sección de Calculados (próximamente)
elif selected_section == 'Calculados':
    st.title("Datos Calculados")
    st.write("Aquí podrás ver los datos MPO calculados basados en modelos o estimaciones.")

# -----------------------------------------------------------------------------
# Sección de Predecidos (próximamente)
elif selected_section == 'Predecidos':
    st.title("Datos Predecidos")
    st.write("Aquí podrás ver los datos MPO predecidos utilizando modelos predictivos.")
