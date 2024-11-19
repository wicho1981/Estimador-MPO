import streamlit as st
import pandas as pd
import plotly.express as px
import io

def mostrar_dashboard(mpo_data_2024, mpo_data_calculado):
    st.title("Dashboard: Comparación de MPO Histórico y Calculado para 2024")
    st.write("Esta sección permite comparar el MPO histórico y el MPO calculado para un día seleccionado de 2024.")

    # Obtener las fechas comunes entre los datos históricos y calculados de 2024
    unique_dates_hist = mpo_data_2024['Fecha'].dt.date.unique()
    unique_dates_calculado = mpo_data_calculado['Fecha'].dt.date.unique()
    common_dates = list(set(unique_dates_hist).intersection(set(unique_dates_calculado)))
    common_dates.sort()

    # Verificar si hay fechas comunes
    if not common_dates:
        st.warning("No hay fechas comunes entre los datos históricos y calculados para 2024.")
        return

    # Selección de fecha para la comparación
    selected_date_dashboard = st.selectbox('Selecciona la fecha para la comparación:', common_dates)

    # Filtrar datos históricos y calculados para la fecha seleccionada
    filtered_mpo_historico = mpo_data_2024[mpo_data_2024['Fecha'].dt.date == selected_date_dashboard]
    filtered_mpo_calculado = mpo_data_calculado[mpo_data_calculado['Fecha'].dt.date == selected_date_dashboard]

    # Transformar datos históricos y calculados para la gráfica
    long_data_historico = filtered_mpo_historico.melt(
        id_vars=['Fecha'], value_vars=[str(i) for i in range(24)], var_name='Hora', value_name='MPO'
    )
    long_data_historico['Hora'] = pd.to_numeric(long_data_historico['Hora'])
    long_data_calculado = filtered_mpo_calculado[['Hora', 'Precio ajustado']].copy()
    long_data_calculado.rename(columns={'Precio ajustado': 'MPO'}, inplace=True)
    long_data_calculado['Fuente'] = 'Calculado'
    long_data_historico['Fuente'] = 'Histórico'

    # Combinar datos para la gráfica
    combined_data = pd.concat([long_data_historico, long_data_calculado], ignore_index=True)

    # Mostrar gráfica de comparación
    if combined_data.empty:
        st.warning("No hay datos disponibles para la fecha seleccionada. Por favor, elige otra fecha.")
    else:
        comparison_chart = px.line(combined_data, x='Hora', y='MPO', color='Fuente',
                                   title=f"Comparación del MPO Histórico y Calculado - {selected_date_dashboard} (2024)",
                                   height=600)
        st.plotly_chart(comparison_chart, use_container_width=True)

        # Calcular y mostrar diferencias
        merged_data = pd.merge(
            long_data_historico[['Hora', 'MPO']],
            long_data_calculado[['Hora', 'MPO']],
            on='Hora', suffixes=('_Historico', '_Calculado')
        )
        merged_data['Diferencia'] = merged_data['MPO_Historico'] - merged_data['MPO_Calculado']
        st.dataframe(merged_data, use_container_width=True)

        # Botones para descargar en CSV y Excel
        output_csv_diferencias = merged_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar diferencias como CSV", data=output_csv_diferencias,
                           file_name=f"diferencias_mpo_{selected_date_dashboard}.csv", mime="text/csv")

        output_excel_diferencias = io.BytesIO()
        with pd.ExcelWriter(output_excel_diferencias, engine='xlsxwriter') as writer:
            merged_data.to_excel(writer, index=False, sheet_name='Diferencias')
            worksheet = writer.sheets['Diferencias']
            for i, col in enumerate(merged_data.columns):
                max_len = max(merged_data[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
        output_excel_diferencias.seek(0)
        st.download_button("📥 Descargar diferencias como Excel", data=output_excel_diferencias,
                           file_name=f"diferencias_mpo_{selected_date_dashboard}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
