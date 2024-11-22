import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from io import BytesIO

def mostrar_predecidos():
    st.title("MPO Prediction for a Full Day")
    
    # Year selection (only 2025)
    year = 2025
    file_path = f"data/predicciones_mpo_{year}.csv"

    try:
        pred_key = f"predictions_{year}"  # Key to store in `st.session_state`

        # Check if predictions are already loaded in `st.session_state`
        if pred_key not in st.session_state:
            predicciones_df = pd.read_csv(file_path)
            predicciones_df['FechaHora'] = pd.to_datetime(predicciones_df['FechaHora'])
            st.session_state[pred_key] = predicciones_df  # Save to session
        else:
            predicciones_df = st.session_state[pred_key]

        # Get available dates in the predictions file
        available_dates = predicciones_df['FechaHora'].dt.date.unique()

        # Specific day selection within the chosen year
        prediction_date = st.date_input(
            "Select the day to view MPO:",
            min_value=min(available_dates),
            max_value=max(available_dates),
            value=min(available_dates)
        )
        
        # Filter predictions for the selected day
        day_predictions = predicciones_df[predicciones_df['FechaHora'].dt.date == prediction_date]

        if not day_predictions.empty:
            # Prepare data for the chart and table
            predictions_table = day_predictions[['FechaHora', 'Prediccion_MPO']].copy()
            predictions_table['Hour'] = predictions_table['FechaHora'].dt.hour
            predictions_table = predictions_table[['Hour', 'Prediccion_MPO']]

            # Remove index numbers for the table
            predictions_table.index = [''] * len(predictions_table)

            # Display hourly prediction chart for the selected day using Plotly
            st.subheader(f"MPO Predictions for {prediction_date.strftime('%Y-%m-%d')}")
            fig = px.line(
                predictions_table,
                x='Hour',
                y='Prediccion_MPO',
                title=f"MPO Predictions for {prediction_date.strftime('%Y-%m-%d')}",
                labels={'Hour': 'Hour', 'Prediccion_MPO': 'Predicted MPO (COP/kWh)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display table with hours and predicted MPOs
            st.subheader("Prediction Data")
            st.dataframe(predictions_table)

            # Buttons for downloading the data
            st.subheader("Download Prediction Data")
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])  # Create columns to center buttons

            with col2:
                # CSV Download Button
                csv = predictions_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"predictions_{prediction_date.strftime('%Y-%m-%d')}.csv",
                    mime='text/csv'
                )

            with col3:
                # Excel Download Button
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    predictions_table.to_excel(writer, index=False, sheet_name='Predictions')
                excel_data = output.getvalue()
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name=f"predictions_{prediction_date.strftime('%Y-%m-%d')}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        else:
            st.warning("No predictions found for the selected date.")
    except FileNotFoundError:
        st.error(f"Predictions file for the year {year} not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the predictions: {e}")
