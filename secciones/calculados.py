import streamlit as st
import plotly.express as px
import pandas as pd
from io import BytesIO

def mostrar_calculados(mpo_data_calculado):
    st.title("Calculated MPO Data")
    st.write("Visualization of adjusted calculated MPO data.")

    # Obtener las fechas únicas para el selector
    unique_dates = mpo_data_calculado['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Select a date:', unique_dates)

    # Filtrar datos según la fecha seleccionada
    filtered_data = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date]

    # Mostrar solo las columnas 'Hora' y 'Precio ajustado'
    display_data = filtered_data[['Hora', 'Precio ajustado']].copy()

    # Hacer el índice invisible
    display_data.index = [''] * len(display_data)

    # Crear gráfica con el eje Y etiquetado con la unidad
    fig = px.line(
        display_data,
        x='Hora',
        y='Precio ajustado',
        title=f"Calculated MPO Data - {selected_date}",
        labels={'Precio ajustado': 'Calculated MPO (COP/kWh)'}  # Etiqueta personalizada
    )
    # Mostrar la gráfica primero
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar los datos filtrados en una tabla interactiva
    st.subheader(f"Data for {selected_date}")
    st.dataframe(display_data)

    # Botones para descargar los datos centrados
    st.subheader("Download Filtered Data")
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])  # Crear columnas para centrar botones

    with col2:
        # Botón para descargar como CSV
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"calculated_data_{selected_date}.csv",
            mime='text/csv'
        )

    with col3:
        # Botón para descargar como Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_data.to_excel(writer, index=False, sheet_name='Filtered Data')
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"calculated_data_{selected_date}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
