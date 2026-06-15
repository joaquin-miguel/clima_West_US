import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb


# ── Página ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard - UNSAM",
    #page_icon="⛈️",
    layout="wide"  
)

# HEADER
st.markdown(
    "<div style='margin-bottom:4px;font-size:20px;color:#888;'>"
    f"Datos: Abril 2025 → Mayo 2026",
    unsafe_allow_html=True,
)

# 2. Definir las páginas disponibles
pagina_detalle = st.Page("pages/detalles.py", title="Técnica", icon="📊")
pagina_negocio = st.Page("pages/negocio.py", title="Negocio", icon="🌍")

# 3. Crear el menú de navegación en el panel izquierdo
pg = st.navigation({
    "Secciones del Proyecto": [pagina_detalle, pagina_negocio]
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem !important; }
  [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #888; }
  [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
  div[data-testid="metric-container"] {
    background: #f7f7f5; border-radius: 10px; padding: 14px 18px;
  }
  h1 { font-size: 1.6rem !important; font-weight: 500 !important; }
  h2 { font-size: 1rem !important; font-weight: 500 !important;
       text-transform: uppercase; letter-spacing: .07em; color: #888; }
</style>
""", unsafe_allow_html=True)

# 5. Ejecutar la navegación
pg.run()



# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
#st.caption(
#    "Datos: Open-Meteo · Modelo: XGBoost · "
#)
