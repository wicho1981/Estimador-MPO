import streamlit as st
import pandas as pd
from datetime import datetime

def mostrar_predecidos():
    st.title("Predicción del MPO para un Día Completo")
    
    # Selección de año
    año = st.selectbox("Selecciona el año de las predicciones:", [2016, 2024, 2025])
    file_path = f"data/predicciones_mpo_{año}.csv"

    try:
        # Cargar predicciones guardadas para el año seleccionado
        predicciones_df = pd.read_csv(file_path)
        predicciones_df['FechaHora'] = pd.to_datetime(predicciones_df['FechaHora'])
        
        # Obtener las fechas disponibles en el archivo de predicciones
        fechas_disponibles = predicciones_df['FechaHora'].dt.date.unique()
        
        # Selección de día específico dentro del año seleccionado
        fecha_prediccion = st.date_input("Selecciona el día para visualizar el MPO:", min_value=min(fechas_disponibles), max_value=max(fechas_disponibles))
        
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

            # Mostrar comparación con datos históricos si es un día de febrero de 2016 o 2024
            if fecha_prediccion.month == 2 and año in [2016, 2024]:
                historico_file_path = f"data/Dataset{año}.csv"
                historico_df = pd.read_csv(historico_file_path, delimiter=';', decimal='.', encoding='ISO-8859-1')
                historico_df['Fecha'] = pd.to_datetime(historico_df['Fecha'], dayfirst=True)
                
                # Filtrar datos históricos para el día seleccionado
                historico_dia = historico_df[historico_df['Fecha'].dt.date == fecha_prediccion]
                
                if not historico_dia.empty:
                    # Formatear datos históricos para visualización
                    historico_long = historico_dia.melt(id_vars=['Fecha'], var_name='Hora', value_name='MPO_Historico')
                    historico_long['Hora'] = pd.to_numeric(historico_long['Hora'], errors='coerce')
                    historico_long.dropna(subset=['MPO_Historico'], inplace=True)
                    
                    # Crear columna de FechaHora para alineación
                    historico_long['FechaHora'] = pd.to_datetime(historico_long['Fecha'].dt.date.astype(str) + ' ' + historico_long['Hora'].astype(int).astype(str) + ':00:00')
                    
                    # Fusionar datos históricos y predicciones para comparación
                    comparacion_df = pd.merge(predicciones_dia[['FechaHora', 'Prediccion_MPO']], historico_long[['FechaHora', 'MPO_Historico']], on='FechaHora', how='left')
                    
                    # Crear tabla con solo hora y valores de MPO para comparación
                    comparacion_df['Hora'] = comparacion_df['FechaHora'].dt.hour
                    comparacion_df = comparacion_df[['Hora', 'Prediccion_MPO', 'MPO_Historico']]
                    
                    # Mostrar gráfico comparativo de MPO predicho y histórico
                    st.subheader("Comparación de MPO Predicho y Histórico")
                    st.line_chart(comparacion_df.set_index("Hora")[["Prediccion_MPO", "MPO_Historico"]], use_container_width=True)
                    st.write(comparacion_df)
                else:
                    st.warning("No se encontraron datos históricos para la fecha seleccionada.")
            else:
                st.write("Predicciones para el día seleccionado (sin datos históricos para comparación).")
        else:
            st.warning("No se encontraron predicciones para la fecha seleccionada.")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de predicciones para el año {año}.")
