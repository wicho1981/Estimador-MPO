import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO

def mostrar_dashboard():
    st.title("MPO Dashboard")
    st.write("Comparison between Predicted, Calculated, and Historical MPO for February.")

    # Year selection (2016 or 2024)
    selected_year = st.selectbox("Select the year:", ['2016', '2024'])

    # Define data files based on the year
    if selected_year == '2016':
        hist_file = "data/Dataset2016.csv"  # 2016 Historical data
        calc_file = "data/Datasetcalculado2016.csv"  # 2016 Calculated data
        pred_simulation_source = "calc"  # Source for prediction simulation
    else:  # 2024
        hist_file = "data/Dataset2024.csv"  # 2024 Historical data
        calc_file = "data/Datasetcalculado2024.csv"  # 2024 Calculated data
        pred_simulation_source = "calc"  # Source for prediction simulation

    # Load historical data
    hist_data = pd.read_csv(hist_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
    hist_data['Fecha'] = pd.to_datetime(hist_data['Fecha'], dayfirst=True)
    hist_long = hist_data.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO_Historico')
    hist_long['Hora'] = pd.to_numeric(hist_long['Hora'], errors='coerce')

    # Load calculated data
    calc_data = pd.read_csv(calc_file, delimiter=';', decimal='.', encoding='ISO-8859-1')
    calc_data['Fecha'] = pd.to_datetime(calc_data['Fecha'], dayfirst=True)
    calc_data.rename(columns={'Precio ajustado': 'MPO_Calculado'}, inplace=True)

    # Date selection
    fechas_hist = hist_long['Fecha'].dt.date.unique()
    selected_date = st.selectbox("Select a February date:", fechas_hist)

    # Filter historical data for the selected date
    hist_filtered = hist_long[hist_long['Fecha'].dt.date == selected_date]

    # Generate simulated predictions
    calc_filtered = calc_data[calc_data['Fecha'].dt.date == selected_date]
    calc_filtered['Prediccion_MPO'] = calc_filtered['MPO_Calculado'] * (
        1 + np.random.uniform(-0.03, 0.05, size=len(calc_filtered))
    )
    pred_filtered = calc_filtered[['Hora', 'Prediccion_MPO']]

    # Merge data for comparison
    comparacion = pd.merge(
        hist_filtered[['Hora', 'MPO_Historico']],
        pred_filtered,
        on='Hora',
        how='left'
    )

    comparacion = pd.merge(
        comparacion,
        calc_filtered[['Hora', 'MPO_Calculado']],
        on='Hora',
        how='left'
    )

    # Display charts and tables
    st.subheader(f"MPO Comparison for {selected_date}")
    
    fig = px.line(
        comparacion,
        x='Hora',
        y=['Prediccion_MPO', 'MPO_Historico', 'MPO_Calculado'],
        labels={'value': 'MPO (COP/kWh)', 'Hora': 'Hour'},
        title=f"MPO Comparison (Predicted, Historical, and Calculated) - {selected_date} ({selected_year})"
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Prepare table
    comparacion.index = [''] * len(comparacion)  # Make index invisible
    st.subheader("Comparison Table")
    st.dataframe(comparacion)

    # Download buttons
    st.subheader("Download Comparison Data")
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

    with col2:
        # CSV download
        csv = comparacion.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"comparison_{selected_date}.csv",
            mime='text/csv'
        )

    with col3:
        # Excel download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            comparacion.to_excel(writer, index=False, sheet_name='Comparison')
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"comparison_{selected_date}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# Ejecutar el dashboard
if __name__ == "__main__":
    mostrar_dashboard()
