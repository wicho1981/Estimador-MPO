import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

def mostrar_historicos(mpo_data_2016, mpo_data_2024):
    st.title("Historical MPO Data")
    st.write("Visualization of historical MPO data.")
    
    # Selección de año
    selected_year = st.selectbox('Select the year:', ['2016', '2024'])
    mpo_data_df = mpo_data_2016 if selected_year == '2016' else mpo_data_2024

    # Selección de fecha dentro del año seleccionado
    unique_dates = mpo_data_df['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Select a date:', unique_dates)
    
    # Filtrar datos según la fecha seleccionada
    filtered_data = mpo_data_df[mpo_data_df['Fecha'].dt.date == selected_date].copy()

    # Convertir a formato largo para graficar (si hay columnas "Hora")
    long_data = filtered_data.melt(id_vars=['Fecha'], var_name='Hour', value_name='MPO')
    long_data['Hour'] = pd.to_numeric(long_data['Hour'], errors='coerce')

    # Filtrar columnas para la tabla
    table_data = long_data[['Hour', 'MPO']].dropna()

    # Crear gráfica con etiqueta personalizada
    fig = px.line(
        table_data,
        x='Hour',
        y='MPO',
        title=f"Historical MPO Data - {selected_date} ({selected_year})",
        labels={'MPO': 'Historical MPO (COP/kWh)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar los datos filtrados en una tabla interactiva
    st.subheader(f"Data for {selected_date} ({selected_year})")
    
    # Hacer el índice "invisible"
    table_data.index = [''] * len(table_data)
    st.dataframe(table_data)

    # Botones para descargar los datos centrados
    st.subheader("Download Filtered Data")
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])  # Crear columnas para centrar botones

    with col2:
        # Botón para descargar como CSV
        csv = table_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"historical_data_{selected_date}_{selected_year}.csv",
            mime='text/csv'
        )

    with col3:
        # Botón para descargar como Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            table_data.to_excel(writer, index=False, sheet_name='Filtered Data')
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"historical_data_{selected_date}_{selected_year}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
