# pages/1_📊_Resumen.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Importamos la lógica y variables de nuestro propio archivo utils
from src.utils import (
    compute_shap_global, get_global_mae_comparison, compute_global_metrics,
    COLOR_CORRECTED, CATS, PATH_MODELO_XGB, PATH_MODELO_CAT, PATH_PARQUET_XGB, PATH_PARQUET_CAT, PATH_SHAP_XGB, PATH_SHAP_CAT
)

# Calculamos SHAP para ambos modelos
df_shap_xgb = compute_shap_global(PATH_SHAP_XGB, PATH_MODELO_XGB, tipo="xgboost")
df_shap_cat = compute_shap_global(PATH_SHAP_CAT, PATH_MODELO_CAT, tipo="catboost")

# Ahora debe calcular el error para: orig, xgb y cat
df_metricas = compute_global_metrics(PATH_PARQUET_XGB, PATH_PARQUET_CAT)

# ==========================================
# PANEL IZQUIERDO (FILTROS FIJOS)
# ==========================================
#st.sidebar.markdown("### Panel de Control")

# Filtro fijo como botones (pills) 
#filtro_altura = st.sidebar.pills(
#    "Filtro de Altitud",
#    options=CATS,
#    default=CATS,
#    selection_mode="multi"
#)

# Si el usuario deselecciona todos, mostramos todos por defecto
#if not filtro_altura:
#    filtro_altura = CATS


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN — KPIs
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Métricas globales")

pct = lambda new, old: abs((new - old) / abs(old) * 100)

col_orig, col_cat, col_xgb = st.columns(3)


 
with col_orig:
    st.markdown(
        """
        <div style="
            background-color:#888780;
            padding:10px;
            border-radius:5px;
            color:white;
            font-weight:bold;
        ">
            Open-Meteo (Original)
        </div>
        """,
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    c1.metric("BIAS",  f"{abs(df_metricas['bias_orig']):+.2f} °C")
    c2.metric("MAE",   f"{df_metricas['mae_orig']:.2f} °C")


with col_cat:
    #st.warning("**Modelo CatBoost**") # Puedes usar el color de acento que prefieras
    st.markdown(
        """
        <div style="
            background-color:#378ADD;
            padding:10px;
            border-radius:5px;
            color:white;
            font-weight:bold;
        ">
            Modelo CatBoost
        </div>
        """,
        unsafe_allow_html=True
    )
    c1, c2= st.columns(2)   # se puede agregar RMSE
    c1.metric("BIAS",  f"{df_metricas['bias_cat']:+.2f} °C")
    c2.metric("MAE",   f"{df_metricas['mae_cat']:.2f} °C",
              delta=f"{pct(df_metricas['mae_cat'],  df_metricas['mae_orig']):.0f}% de mejora")
    
with col_xgb:
    #st.success("**Modelo XGBoost**")
    st.markdown(
        """
        <div style="
            background-color:#D85A30;
            padding:10px;
            border-radius:5px;
            color:white;
            font-weight:bold;
        ">
            Modelo XGBoost
        </div>
        """,
        unsafe_allow_html=True
    )
    
    c1, c2 = st.columns(2)
    c1.metric("BIAS",  f"{df_metricas['bias_xgb']:+.2f} °C"
             #,delta=f"{pct(df_metricas['bias_xgb'], df_metricas['bias_orig']):.0f}%",  delta_color="inverse"
            )
    c2.metric("MAE",   f"{df_metricas['mae_xgb']:.2f} °C",
              delta=f"{pct(df_metricas['mae_xgb'],  df_metricas['mae_orig']):.0f}% de mejora")



df_comp = get_global_mae_comparison(PATH_PARQUET_XGB, PATH_PARQUET_CAT)

# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN — INTERPRETABILIDAD SHAP
# ═══════════════════════════════════════════════════════════════════════════
#st.markdown("---")
#st.markdown("## Interpretabilidad: ¿Qué miran los modelos?")
#st.caption("Comparación de Feature Importance global. Observa cómo CatBoost asigna mayor peso a variables categóricas como el ID de la Estación.")
st.markdown('')
st.markdown('')

col_s1, col_s2 = st.columns(2)



# BARRAS MODELO
df_plot_cat = df_shap_cat.sort_values("shap_mean", ascending=False)

porcentajes_cat = (
    df_plot_cat["shap_mean"]
    / df_plot_cat["shap_mean"].sum()
    * 100
) 

with col_s1:
    #st.plotly_chart(fig, use_container_width=True)
    #fig_xgb = go.Figure(go.Bar(
    #    x=df_shap_cat["shap_mean"].round(4),
    #    y=df_shap_cat["feature"],
    #    orientation="h",
    #    marker_color="#378ADD", 
    #    marker_line_width=0,
    #))
    #fig_xgb.update_layout(
    #    title_text="SHAP — CatBoost", title_font_size=16,
    #    xaxis_title="Impacto medio en la predicción",
    #    margin=dict(l=0, r=0, t=40, b=0), height=380,
    #    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    #    yaxis={'categoryorder':'total ascending'} # Asegura orden de importancia
    #)
    #st.plotly_chart(fig_xgb, use_container_width=True)
    
    # ----
    max_x = porcentajes_cat.max()

    fig_xgb = go.Figure(
        go.Bar(
            x=porcentajes_cat,
            y=df_plot_cat["feature"],
            orientation="h",
            marker_color="#378ADD",
            text=[f"{v:.1f}%" for v in porcentajes_cat],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig_xgb.update_layout(
        title="Importancia de Features (SHAP) - CatBoost",
        xaxis_title="Importancia (%)",
        height=400,
        margin=dict(l=0, r=80, t=50, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            autorange="reversed"  # más importante arriba
        ),
    )

    fig_xgb.update_xaxes(
        showgrid=True,
        griddash="dash",
        range=[0, max_x * 1.15]
    )

    st.plotly_chart(fig_xgb, use_container_width=True)
    
    
    
df_plot_xgb = df_shap_xgb.sort_values("shap_mean", ascending=False)

porcentajes = (
    df_plot_xgb["shap_mean"]
    / df_plot_xgb["shap_mean"].sum()
    * 100
)   
# SHAP XGBoost
with col_s2:

    max_x = porcentajes.max()

    fig_xgb = go.Figure(
        go.Bar(
            x=porcentajes,
            y=df_plot_xgb["feature"],
            orientation="h",
            marker_color="#D85A30",
            text=[f"{v:.1f}%" for v in porcentajes],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig_xgb.update_layout(
        title="Importancia de Features (SHAP) - XGBoost",
        xaxis_title="Importancia (%)",
        height=400,
        margin=dict(l=0, r=80, t=50, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            autorange="reversed"  # más importante arriba
        ),
    )

    fig_xgb.update_xaxes(
        showgrid=True,
        griddash="dash",
        range=[0, max_x * 1.15]
    )

    st.plotly_chart(fig_xgb, use_container_width=True)

    

