import streamlit as st
import pandas as pd
from datetime import datetime

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
            # Display hourly prediction chart for the selected day
            st.subheader(f"MPO Predictions for {prediction_date.strftime('%Y-%m-%d')}")
            st.line_chart(day_predictions.set_index("FechaHora")["Prediccion_MPO"], use_container_width=True)
            
            # Table with only hours and predicted MPOs
            predictions_table = day_predictions[['FechaHora', 'Prediccion_MPO']].copy()
            predictions_table['Hour'] = predictions_table['FechaHora'].dt.hour
            predictions_table = predictions_table[['Hour', 'Prediccion_MPO']]
            st.write(predictions_table)
        else:
            st.warning("No predictions found for the selected date.")
    except FileNotFoundError:
        st.error(f"Predictions file for the year {year} not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the predictions: {e}")
