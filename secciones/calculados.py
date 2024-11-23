import streamlit as st
import plotly.express as px
import pandas as pd
from io import BytesIO

def mostrar_calculados(mpo_data_calculado2024, mpo_data_calculado2016):
    st.title("Calculated MPO Data")
    st.write(
        """
        In this section, you can visualize the adjusted offered price information 
        under the methodology proposed by resolution project CREG 701 049 2024
        for February 2024 and 2016 dates.
        """
    )

    # Selection of year
    selected_year = st.selectbox("Select the year:", ['2016', '2024'])
    mpo_data_calculado = mpo_data_calculado2016 if selected_year == '2016' else mpo_data_calculado2024

    # Get unique dates for the selector
    unique_dates = mpo_data_calculado['Fecha'].dt.date.unique()
    selected_date = st.selectbox('Select a date:', unique_dates)

    # Filter data based on the selected date
    filtered_data = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date]

    # Display only the columns 'Hora' and 'Precio ajustado'
    display_data = filtered_data[['Hora', 'Precio ajustado']].copy()

    # Make the index invisible
    display_data.index = [''] * len(display_data)

    # Create a graph with the Y-axis labeled with the unit
    fig = px.line(
        display_data,
        x='Hora',
        y='Precio ajustado',
        title=f"Calculated MPO Data - {selected_date} ({selected_year})",
        labels={'Precio ajustado': 'Calculated MPO (COP/kWh)'}  # Custom label
    )
    # Show the graph first
    st.plotly_chart(fig, use_container_width=True)

    # Show filtered data in an interactive table
    st.subheader(f"Data for {selected_date}")
    st.dataframe(display_data)

    # Buttons to download the centered data
    st.subheader("Download Filtered Data")
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])  # Create columns to center buttons

    with col2:
        # Button to download as CSV
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"calculated_data_{selected_date}_{selected_year}.csv",
            mime='text/csv'
        )

    with col3:
        # Button to download as Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_data.to_excel(writer, index=False, sheet_name='Filtered Data')
        excel_data = output.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"calculated_data_{selected_date}_{selected_year}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
