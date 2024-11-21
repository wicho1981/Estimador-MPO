import streamlit as st
import plotly.express as px

def mostrar_calculados(mpo_data_calculado):
    st.title("Calculated MPO Data")
    st.write("Visualization of adjusted calculated MPO data.")

    unique_dates = mpo_data_calculado['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Select a date:', unique_dates)

    filtered_data = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date]
    fig = px.line(filtered_data, x='Hora', y='Precio ajustado', title=f"Calculated MPO Data - {selected_date}")
    st.plotly_chart(fig, use_container_width=True)
