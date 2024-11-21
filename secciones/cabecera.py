import streamlit as st
from pathlib import Path

def mostrar_cabecera():
    # Configurar título y divisor

    st.markdown("---")

    # Configuración de columnas para el logo y los textos
    col1, col2, col3 = st.columns([1, 0.1, 3])

    # Columna 1: Logo
    with col1:
        logo_path = Path(__file__).parent.parent / 'logo.png'  # Ruta ajustada
        if logo_path.is_file():
            st.image(str(logo_path), use_column_width=True)
        else:
            st.error("Error: Logo no encontrado en la ruta especificada.")

    # Columna 2: Línea vertical (usando HTML para el estilo)
    with col2:
        st.markdown("<div style='height: 100%; width: 2px; background-color: #cccccc;'></div>", unsafe_allow_html=True)

    # Columna 3: Título y subtítulo alineados a la derecha
    with col3:
        st.markdown(
            """
            <div style='text-align: right;'>
                <h2 style='margin-bottom: 5px; color: #333333; font-size: 1.6em;'>Estimation of the Maximum Offered Price in the Colombian Wholesale Energy Market Supported by AI</h2>
                <p style='margin-top: 0px; color: #555555; font-size: 1.2em; font-style: italic;'>Authors: Cristian Noguera & Jaider Sanchez</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Divisor inferior
    st.markdown("---")