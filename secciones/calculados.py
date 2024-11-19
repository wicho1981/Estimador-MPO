import streamlit as st
import plotly.express as px

def mostrar_calculados(mpo_data_calculado):
    st.title("Calculados")
    st.write("Visualización de datos calculados ajustados del MPO.")

    unique_dates = mpo_data_calculado['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Selecciona una fecha:', unique_dates)

    filtered_data = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date]
    fig = px.line(filtered_data, x='Hora', y='Precio ajustado', title=f"Datos calculados de MPO - {selected_date}")
    st.plotly_chart(fig, use_container_width=True)
