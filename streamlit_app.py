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
selected_section = st.selectbox('Selecciona la sección:', ['Dashboard','Históricos', 'Calculados', 'Predecidos'])

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
# Sección de Calculados
elif selected_section == 'Calculados':
    st.title("Datos Calculados")
    st.write("Esta sección muestra los datos calculados de MPO ajustado para un día específico y permite filtrar un rango de días para análisis tabular.")

    # Cargar el CSV de calculados
    mpo_data_calculado = get_mpo_data('Datasetcalculado2024.csv')

    # Paso 1: Gráfico - Seleccionar la fecha para una gráfica de un solo día
    st.subheader("Gráfica de líneas para el Precio Ajustado durante un día específico")

    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()  # Asegurarse de mostrar solo fechas únicas
    selected_date_calculado = st.selectbox('Selecciona la fecha para la gráfica (calculados):', unique_dates_calculado)

    # Filtrar los datos para el día seleccionado
    filtered_mpo_data_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_calculado]

    # Convertir los datos a formato largo para graficar cada hora como un punto separado
    long_data_calculado = filtered_mpo_data_calculado[['Fecha', 'Hora', 'Precio ajustado']].copy()

    # Convertir la columna 'Hora' a numérico y eliminar datos no válidos
    long_data_calculado['Hora'] = pd.to_numeric(long_data_calculado['Hora'], errors='coerce')
    long_data_calculado['Precio ajustado'] = pd.to_numeric(long_data_calculado['Precio ajustado'], errors='coerce')

    # Eliminar filas con datos faltantes o no válidos
    long_data_calculado.dropna(subset=['Precio ajustado'], inplace=True)

    # Formatear la fecha como 'dd/mm/aaaa'
    long_data_calculado['Fecha'] = long_data_calculado['Fecha'].dt.strftime('%d/%m/%Y')

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

    # Paso 2: Tabla - Seleccionar un rango de fechas para el análisis tabular
    st.subheader("Vista tabular de los datos calculados para un rango de fechas")

    # Filtrar por rango de fechas
    min_date_calculado = mpo_data_calculado['Fecha'].min().date()
    max_date_calculado = mpo_data_calculado['Fecha'].max().date()

    # Seleccionar rango de fechas para la tabla
    selected_date_range_calculado = st.date_input(
        'Selecciona el rango de fechas para la tabla (calculados):',
        value=[min_date_calculado, max_date_calculado],
        min_value=min_date_calculado,
        max_value=max_date_calculado
    )

    # Convertir las fechas seleccionadas a formato datetime
    selected_start_date_calculado = pd.to_datetime(selected_date_range_calculado[0])
    selected_end_date_calculado = pd.to_datetime(selected_date_range_calculado[1])

    # Filtrar los datos basados en el rango de fechas seleccionado
    filtered_table_data_calculado = mpo_data_calculado[
        (mpo_data_calculado['Fecha'] >= selected_start_date_calculado) &
        (mpo_data_calculado['Fecha'] <= selected_end_date_calculado)
    ]

    # Mostrar la tabla filtrada
    if filtered_table_data_calculado.empty:
        st.warning("No hay datos disponibles para el rango de fechas seleccionado. Por favor, elige otro rango.")
    else:
        # Formatear las fechas como 'dd/mm/aaaa'
        filtered_table_data_calculado['Fecha'] = filtered_table_data_calculado['Fecha'].dt.strftime('%d/%m/%Y')
        
        st.write(f"Datos filtrados desde el {selected_start_date_calculado.strftime('%d/%m/%Y')} hasta el {selected_end_date_calculado.strftime('%d/%m/%Y')}:")
        st.dataframe(filtered_table_data_calculado[['Fecha', 'Hora', 'Precio ajustado']], use_container_width=True)

        # Botón para exportar los datos a Excel
        output_excel_calculado = io.BytesIO()
        with pd.ExcelWriter(output_excel_calculado, engine='xlsxwriter') as writer:
            filtered_table_data_calculado.to_excel(writer, index=False, sheet_name='Datos Calculados Filtrados')

            # Ajustar el ancho de las columnas
            worksheet_calculado = writer.sheets['Datos Calculados Filtrados']
            for i, col in enumerate(filtered_table_data_calculado.columns):
                column_len = filtered_table_data_calculado[col].astype(str).str.len().max()
                column_len = max(column_len, len(col)) + 2
                worksheet_calculado.set_column(i, i, column_len)

        output_excel_calculado.seek(0)

        st.download_button(
            label="📥 Descargar datos calculados filtrados como Excel",
            data=output_excel_calculado,
            file_name="datos_calculados_MPO_2024.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Botón para exportar los datos a CSV
        output_csv_calculado = filtered_table_data_calculado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos calculados filtrados como CSV",
            data=output_csv_calculado,
            file_name="datos_calculados_MPO_2024.csv",
            mime="text/csv"
        )
# -----------------------------------------------------------------------------
# Sección de Predecidos (próximamente)
elif selected_section == 'Predecidos':
    st.title("Datos Predecidos")
    st.write("Aquí podrás ver los datos MPO predecidos utilizando modelos predictivos.")
# -----------------------------------------------------------------------------
# Sección de Dashboard (Comparación de MPO Histórico y Calculado)
elif selected_section == 'Dashboard':
    st.title("Dashboard: Comparación de MPO Histórico y Calculado")
    st.write("Esta sección compara el MPO histórico y el MPO calculado para un día seleccionado del 2024.")

    # Cargar los datos de 2024
    mpo_data_2024 = get_mpo_data('Dataset2024.csv')  # Datos históricos
    mpo_data_calculado = get_mpo_data('Datasetcalculado2024.csv')  # Datos calculados

    # Paso 1: Seleccionar la fecha para comparar
    st.subheader("Comparación del MPO Histórico y Calculado para un día específico")

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
    long_data_historico = pd.melt(
        filtered_mpo_historico,
        id_vars=['Fecha'],  # Mantener la columna Fecha
        value_vars=[str(i) for i in range(24)],  # Seleccionar las columnas de hora 0 a 23
        var_name='Hora',  # La nueva columna se llamará 'Hora'
        value_name='MPO'  # Los valores de cada hora se llamarán 'MPO'
    )

    # Asegurar que la columna 'Hora' es numérica
    long_data_historico['Hora'] = pd.to_numeric(long_data_historico['Hora'])

    # Preparar los datos calculados
    long_data_calculado = filtered_mpo_calculado[['Hora', 'Precio ajustado']].copy()
    long_data_calculado.rename(columns={'Precio ajustado': 'MPO'}, inplace=True)  # Renombrar columna para igualar con histórico
    long_data_calculado['Fuente'] = 'Calculado'  # Añadir una columna para identificar la fuente

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

        # ---------------------------------------------------------------------
        # Calcular las diferencias entre MPO Histórico y Calculado
        st.subheader("Tabla de diferencias entre MPO Histórico y Calculado")

        # Unir los dos dataframes por 'Hora' para calcular las diferencias
        merged_data = pd.merge(long_data_historico[['Hora', 'MPO']], long_data_calculado[['Hora', 'MPO']], on='Hora', suffixes=('_Historico', '_Calculado'))

        # Calcular la diferencia entre el MPO histórico y el calculado
        merged_data['Diferencia'] = merged_data['MPO_Historico'] - merged_data['MPO_Calculado']

        # Mostrar la tabla de diferencias sin la columna del índice
        st.dataframe(merged_data, use_container_width=True)

        # Botón para descargar la tabla de diferencias como CSV
        output_csv_diferencias = merged_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar tabla de diferencias como CSV",
            data=output_csv_diferencias,
            file_name=f"diferencias_mpo_{selected_date_dashboard.strftime('%d_%m_%Y')}.csv",
            mime="text/csv"
        )

        # ---------------------------------------------------------------------
        # Botón para descargar la tabla de diferencias como Excel

        output_excel_diferencias = io.BytesIO()
        with pd.ExcelWriter(output_excel_diferencias, engine='xlsxwriter') as writer:
            # Guardar la tabla sin el índice
            merged_data.to_excel(writer, index=False, sheet_name='Diferencias')

            # Ajustar el ancho de las columnas para mejor legibilidad en Excel
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
