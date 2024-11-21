import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_historicos(mpo_data_2016, mpo_data_2024):
    st.title("Historical MPO Data")
    st.write("Visualization of historical MPO data.")
    
    # Year selection
    selected_year = st.selectbox('Select the year:', ['2016', '2024'])
    mpo_data_df = mpo_data_2016 if selected_year == '2016' else mpo_data_2024

    # Date selection for the selected year
    unique_dates = mpo_data_df['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Select a date:', unique_dates)
    
    # Filter and plot data for the selected date
    filtered_data = mpo_data_df[mpo_data_df['Fecha'].dt.date == selected_date]
    long_data = filtered_data.melt(id_vars=['Fecha'], var_name='Hour', value_name='MPO')
    long_data['Hour'] = pd.to_numeric(long_data['Hour'], errors='coerce')

    fig = px.line(long_data, x='Hour', y='MPO', title=f"Historical MPO Data - {selected_date} ({selected_year})")
    st.plotly_chart(fig, use_container_width=True)
