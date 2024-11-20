import streamlit as st
import pandas as pd
from datetime import datetime

def mostrar_predecidos():
    st.title("Predicción del MPO para un Día Completo")
    
    # Selección de año (solo 2025)
    año = 2025
    file_path = f"data/predicciones_mpo_{año}.csv"

    try:
        pred_key = f"predicciones_{año}"  # Clave para almacenar en `st.session_state`

        # Verificar si las predicciones ya están cargadas en `st.session_state`
        if pred_key not in st.session_state:
            predicciones_df = pd.read_csv(file_path)
            predicciones_df['FechaHora'] = pd.to_datetime(predicciones_df['FechaHora'])
            st.session_state[pred_key] = predicciones_df  # Guardar en la sesión
        else:
            predicciones_df = st.session_state[pred_key]

        # Obtener las fechas disponibles en el archivo de predicciones
        fechas_disponibles = predicciones_df['FechaHora'].dt.date.unique()

        # Selección de día específico dentro del año seleccionado
        fecha_prediccion = st.date_input(
            "Selecciona el día para visualizar el MPO:",
            min_value=min(fechas_disponibles),
            max_value=max(fechas_disponibles),
            value=min(fechas_disponibles)
        )
        
        # Filtrar las predicciones para el día seleccionado
        predicciones_dia = predicciones_df[predicciones_df['FechaHora'].dt.date == fecha_prediccion]

        if not predicciones_dia.empty:
            # Mostrar gráfico de predicciones horarios para el día seleccionado
            st.subheader(f"Predicciones de MPO para el {fecha_prediccion.strftime('%Y-%m-%d')}")
            st.line_chart(predicciones_dia.set_index("FechaHora")["Prediccion_MPO"], use_container_width=True)
            
            # Tabla con solo las horas y los MPO predichos
            tabla_predicciones = predicciones_dia[['FechaHora', 'Prediccion_MPO']].copy()
            tabla_predicciones['Hora'] = tabla_predicciones['FechaHora'].dt.hour
            tabla_predicciones = tabla_predicciones[['Hora', 'Prediccion_MPO']]
            st.write(tabla_predicciones)
        else:
            st.warning("No se encontraron predicciones para la fecha seleccionada.")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de predicciones para el año {año}.")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar las predicciones: {e}")
