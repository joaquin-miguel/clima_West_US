# pages/2_🌍_Mapa.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from src.utils import (cargar_df, 
                       compute_global_metrics, compute_mae_by_category, compute_map_data,compute_diurnal_bias,
                       PATH_PARQUET_XGB, COLOR_ORIGINAL, PATH_SHAP_XGB, CATS, COLORS_ALT, PATH_PARQUET_CAT, PATH_SHAP_CAT
)

import folium
from streamlit_folium import st_folium
from branca.colormap import linear
from branca.colormap import LinearColormap

# ==========================================
# PANEL IZQUIERDO (FILTROS FIJOS)
# ==========================================
st.sidebar.markdown("### Panel de Control")

# Filtro fijo como botones (pills) seleccionables múltiple (Streamlit >= 1.35)
filtro_altitud = st.sidebar.pills(
    "Filtro de Altitud",
    options=CATS,
    default=CATS,
    selection_mode="multi",
)
st.sidebar.markdown("""
Baja: 0–500 m  
Media: 500–1200 m  
Alta: 1200–1600 m  
Muy Alta: >1600 m
""")

modelo_seleccionado = st.sidebar.radio(
    "Modelo",
    ["XGBoost", 'CatBoost'],
    horizontal=True
)

if modelo_seleccionado == "XGBoost":
    PATH_PARQUET = PATH_PARQUET_XGB
    PATH_SHAP = PATH_SHAP_XGB
    NOMBRE_MODELO = "XGBoost"
    TIPO_MODELO = "xgboost"
else:
    PATH_PARQUET = PATH_PARQUET_CAT
    PATH_SHAP = PATH_SHAP_CAT
    NOMBRE_MODELO = "CatBoost"
    TIPO_MODELO = "catboost"


# Si el usuario deselecciona todos, mostramos todos por defecto
if not filtro_altitud:
    selected = CATS
    
if modelo_seleccionado == "XGBoost":
    PATH_PARQUET = PATH_PARQUET_XGB
    PATH_SHAP = PATH_SHAP_XGB
    NOMBRE_MODELO = "XGBoost"
    TIPO_MODELO = "xgboost"
else:
    PATH_PARQUET = PATH_PARQUET_CAT
    PATH_SHAP = PATH_SHAP_CAT
    NOMBRE_MODELO = "CatBoost"
    TIPO_MODELO = "catboost"
    
#=================== DATASET ============================
df_mapa = compute_map_data(PATH_PARQUET)

df_diurno = compute_diurnal_bias(PATH_PARQUET)

 


# ===========================================================
df_filtrado = df_mapa[df_mapa["categoria_altitud"].isin(filtro_altitud)]
lim = max(abs(df_mapa["mejora_pct"].min()),
          abs(df_mapa["mejora_pct"].max()))

# ==================== MAPA CATEG
col1, col2, col3 = st.columns([3, 0.2, 2])
with col1:

    m = folium.Map(
        location=[40.852, -111.968],
        zoom_start=5.1,
        tiles="OpenTopoMap",   # OpenTopoMap <- relieve
        attribution_control=False
    )

    folium.TileLayer(
        #tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Terrain",
        opacity=0.5
    ).add_to(m)

    # Escala Viridis
    colormap = LinearColormap(
        colors=[
            "#440154",
            "#482878",
            "#3E4989",
            "#31688E",
            "#26828E",
            "#1F9E89",
            "#35B779",
            "#6CCE59",
            "#B4DE2C",
            "#FDE725"
        ],
        vmin=0,
        vmax=lim
    )

    colormap.caption = "Mejora %"
    

    for _, row in df_filtrado.iterrows():
        color = colormap(row["mejora_pct"])

        popup = f"""
        <b>Station:</b> {row['station_id']}<br>
        <b>Elevación:</b> {row['elevation']:.0f} m<br>
        <b>MAE Orig:</b> {row['mae_orig']:.2f}<br>
        <b>MAE {NOMBRE_MODELO}:</b> {row['mae_corr']:.2f}<br>
        <b>Mejora:</b> {row['mejora_pct']:.2f}%
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=color,
            fill_color=color,
            fill_opacity=0.3,
            opacity=0.85,
            weight=10,
            popup=popup,
            tooltip=f"""
                {row['station_id']}<br>
                Mejora: {row['mejora_pct']:.1f}%
            """
        ).add_to(m)
        
    legend_html = f"""
        <div style="
        position: fixed;
        top: 50%;
        right: 40px;
        transform: translateY(-50%);
        width: 22px;
        height: 220px;
        background: linear-gradient(
        to top,
        #440154,
        #482878,
        #3E4989,
        #31688E,
        #26828E,
        #1F9E89,
        #35B779,
        #6CCE59,
        #B4DE2C,
        #FDE725
        );
        border:1px solid #888;
        z-index:9999;
        "></div>

        <div style="
        position: fixed;
        top: calc(50% - 135px);
        right: 40px;
        font-size:12px;
        z-index:9999;
        ">
        {lim:.0f}%
        </div>

        <div style="
        position: fixed;
        top: calc(50% + 110px);
        right: 40px;
        font-size:12px;
        z-index:9999;
        ">
        0%
        </div>

        <div style="
        position: fixed;
        top: calc(50% - 160px);
        right: 10px;
        font-size:13px;
        font-weight:bold;
        z-index:9999;
        ">
        Mejora (%)
        </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))


    #colormap.add_to(m)

    st_folium(
        m,
        width=600,
        height=540,
        use_container_width=True
    )  


# ===========================================================
df_mae = compute_mae_by_category(PATH_PARQUET)
df_f = df_mae[df_mae["categoria_altitud"].isin(filtro_altitud)]

# ================= BARRAS
with col3:
    st.markdown("## Análisis por altitud")
    
    fig = go.Figure()
    fig.add_bar(
        name="OpenMeteo",  x=df_f["categoria_altitud"], y=df_f["mae_orig"].round(3),
        marker_color=COLOR_ORIGINAL,  marker_line_width=0,
    )
    fig.add_bar(
        name=NOMBRE_MODELO, x=df_f["categoria_altitud"], y=df_f["mae_corr"].round(3),
        marker_color='#D85A30', marker_line_width=0,
    )
    fig.update_layout(
        title_text="MAE por categoría de altitud", title_font_size=14,
        barmode="group", yaxis_title="MAE (°C)", yaxis_range=[0, None],
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=40, b=10), height=340,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width='stretch')



# ===================================== BIAS TEMPORAL
# Filtramos por categoría seleccionada en el sidebar
df_f = df_diurno[df_diurno["categoria_altitud"].isin(filtro_altitud)]


#col_line = st.columns(1)
# ── MAE original vs corregido por categoría ──────────────────────────────────

#with col_line:
fig = go.Figure()
for cat in filtro_altitud:
    data_cat = df_f[df_f["categoria_altitud"] == cat]
    
    # Línea del error original (lo que fallaba la API)
    fig.add_trace(go.Scatter(
        x=data_cat["hora_utc"], y=data_cat["bias_real"],
        name=f"{cat} - OpenMeteo",
        line=dict(color=COLORS_ALT[cat], dash="dot", width=2)
    ))
    
    # Línea del error corregido (lo que logramos con XGBoost)
    fig.add_trace(go.Scatter(
        x=data_cat["hora_utc"], y=data_cat["bias_corregido"],
        name=f"{cat} - {NOMBRE_MODELO}",
        line=dict(color=COLORS_ALT[cat], width=3)
    ))

# linea en cero
fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="black")

fig.update_layout(
    title_text="Ciclo Diurno del sesgo", title_font_size=14,
    xaxis=dict(tickmode='linear', dtick=3, title="Hora"),
    yaxis_title="BIAS (°C)",
    hovermode="x unified",
    height=400,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, width='stretch') 



    



