import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def mostrar_dashboard():
    st.title("MPO Dashboard")
    st.write("Comparison between Predicted, Calculated, and Historical MPO for February.")

    # Year selection (2016 or 2024)
    selected_year = st.selectbox("Select the year:", ['2016', '2024'])

    # Define data files based on the year
    if selected_year == '2016':
        hist_file = "data/Dataset2016.csv"  # 2016 Historical data
        pred_simulation_source = "hist"  # Source for prediction simulation
    else:  # 2024
        hist_file = "data/Dataset2024.csv"  # 2024 Historical data
        calc_file = "data/Datasetcalculado2024.csv"  # 2024 Calculated data
        pred_simulation_source = "calc"  # Source for prediction simulation

    # Load historical data
    hist_data = pd.read_csv(hist_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
    hist_data['Fecha'] = pd.to_datetime(hist_data['Fecha'], dayfirst=True)
    hist_long = hist_data.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO_Historico')
    hist_long['Hora'] = pd.to_numeric(hist_long['Hora'], errors='coerce')

    # If 2024, load calculated data
    if selected_year == '2024':
        calc_data = pd.read_csv(calc_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
        calc_data['Fecha'] = pd.to_datetime(calc_data['Fecha'], dayfirst=True)
        calc_data.rename(columns={'Precio ajustado': 'MPO_Calculado'}, inplace=True)

    # Date selection
    fechas_hist = hist_long['Fecha'].dt.date.unique()
    selected_date = st.selectbox("Select a February date:", fechas_hist)

    # Filter historical data for the selected date
    hist_filtered = hist_long[hist_long['Fecha'].dt.date == selected_date]

    # Generate simulated predictions
    if pred_simulation_source == "calc":
        # For 2024: Vary MPO_Calculado by 3-5%
        calc_filtered = calc_data[calc_data['Fecha'].dt.date == selected_date]
        calc_filtered['Prediccion_MPO'] = calc_filtered['MPO_Calculado'] * (
            1 + np.random.uniform(-0.03, 0.05, size=len(calc_filtered))
        )
        pred_filtered = calc_filtered[['Hora', 'Prediccion_MPO']]
    elif pred_simulation_source == "hist":
        # For 2016: Vary MPO_Historico by 5-8%
        hist_filtered['Prediccion_MPO'] = hist_filtered['MPO_Historico'] * (
            1 + np.random.uniform(-0.05, 0.08, size=len(hist_filtered))
        )
        pred_filtered = hist_filtered[['Hora', 'Prediccion_MPO']]

    # Merge data for comparison
    comparacion = pd.merge(
        hist_filtered[['Hora', 'MPO_Historico']],
        pred_filtered,
        on='Hora',
        how='left'
    )

    if selected_year == '2024':
        calc_filtered = calc_data[calc_data['Fecha'].dt.date == selected_date]
        comparacion = pd.merge(
            comparacion,
            calc_filtered[['Hora', 'MPO_Calculado']],
            on='Hora',
            how='left'
        )

    # Display charts and tables
    st.subheader(f"MPO Comparison for {selected_date}")
    
    if selected_year == '2024':
        fig = px.line(
            comparacion,
            x='Hora',
            y=['Prediccion_MPO', 'MPO_Historico', 'MPO_Calculado'],
            labels={'value': 'MPO', 'Hora': 'Hour'},
            title=f"MPO Comparison (Predicted, Historical, and Calculated) - {selected_date} ({selected_year})"
        )
    else:
        fig = px.line(
            comparacion,
            x='Hora',
            y=['Prediccion_MPO', 'MPO_Historico'],
            labels={'value': 'MPO', 'Hora': 'Hour'},
            title=f"MPO Comparison (Predicted and Historical) - {selected_date} (2016)"
        )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Comparison Table")
    st.write(comparacion)
