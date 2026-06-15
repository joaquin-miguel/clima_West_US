import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from catboost import CatBoostRegressor

# ═══════════════════════════════════════════════════════════════════════════
# PATHS — ajustá si tu estructura de carpetas es distinta
# ═══════════════════════════════════════════════════════════════════════════
 
PATH_PARQUET_XGB = "./data/df_test_xgb_t1.parquet"
PATH_PARQUET_CAT = "./data/df_test_catboost_t1.parquet"

PATH_MODELO_XGB  = "./models/modelo_xgb_t1.json"
PATH_MODELO_CAT = "./models/modelo_catboost_t1.json"

PATH_SHAP_XGB   = "./models/shap_values_xgb_t1.pkl"
PATH_SHAP_CAT   = "./models/shap_values_catboost_t1.pkl"

# ── Paleta — 4 categorías ────────────────────────────────────────────────────
COLOR_ORIGINAL  = "#888780"
COLOR_CORRECTED = "#378ADD"
COLORS_ALT = {
    "Baja"     : "#888780", # 0-500m
    "Media"    : "#378ADD",  # 500-1200m
    "Alta"     : "#1D9E75", # 1200-1600m
    "Muy Alta" : "#D85A30", # 1600-4000m
}
CATS = list(COLORS_ALT.keys())  # orden canónico
 
# ── Columnas reales de tu parquet ────────────────────────────────────────────
# temperatura_o  = observación de la estación (target real)
# temperature_p  = predicción Open-Meteo original
# temp_error     = temperature_p - temperatura_o  (ya calculado en tu df)
# categoria_altitud = Baja / Media / Alta / Alpina
# fecha          = datetime64[us, UTC]
 
COL_OBS       = "temperature_o"
COL_PRED_ORIG = "temperature_p"
COL_ERROR     = "temp_error"          # bias instantáneo
COL_ERROR_ABS = "temp_error_absoluto" # |error| para MAE
COL_CATEGORIA = "categoria_altitud"

# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS — reemplazá con tus DataFrames reales
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Cargando dataset (~4.5 M filas)…")
def cargar_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
 
 
@st.cache_resource(show_spinner="Cargando modelo XGBoost…")
def cargar_modelo(path, tipo="xgboost"):
    """
    Carga el modelo pre-entrenado dependiendo del tipo.
    tipo: 'xgboost' o 'catboost'
    """
    if tipo == "xgboost":
        modelo = xgb.XGBRegressor() # O el estimador que uses
        modelo.load_model(path)
        return modelo
        
    elif tipo == "catboost":
        modelo = CatBoostRegressor()
        modelo.load_model(path)
        return modelo
        
    else:
        raise ValueError("Tipo de modelo no soportado. Usa 'xgboost' o 'catboost'.")
 
 
@st.cache_resource(show_spinner="Cargando SHAP values…")
def cargar_shap(path: str) -> np.ndarray:
    # cache_resource porque numpy arrays grandes no son hashables
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════
# AGREGADOS — calculados sobre df_test, sin predict en runtime
# ═══════════════════════════════════════════════════════════════════════════
 
@st.cache_data(show_spinner=False)
def compute_global_metrics(path_xgb: str, path_cat: str) -> dict:
    # Cargamos ambos datasets
    df_xgb = cargar_df(path_xgb)
    df_cat = cargar_df(path_cat)
    
    # El error original de Open-Meteo es el mismo en ambos, lo sacamos del XGB
    return {
        "bias_orig" : float(df_xgb["bias_orig"].mean()),
        "mae_orig"  : float(df_xgb["error_orig_abs"].mean()),
        
        # Métricas de XGBoost
        "bias_xgb"  : float(df_xgb["bias_corr"].mean()),
        "mae_xgb"   : float(df_xgb["error_corr_abs"].mean()),
        
        # Métricas de CatBoost
        "bias_cat"  : float(df_cat["bias_corr"].mean()),
        "mae_cat"   : float(df_cat["error_corr_abs"].mean()),
    }
 
 
@st.cache_data(show_spinner=False)
def compute_mae_by_category(path: str) -> pd.DataFrame:
    df = cargar_df(path)
    result = (
        df.groupby("categoria_altitud", observed=True)
        .agg(
            mae_orig=("error_orig_abs", "mean"),
            mae_corr=("error_corr_abs", "mean"),
            n=("error_orig_abs", "count"),
        )
        .reindex(CATS)
        .reset_index()
    )
    return result
 
 
@st.cache_data(show_spinner=False)
def compute_bias_by_hour(path: str) -> pd.DataFrame:
    df = cargar_df(path)
    result = (
        df.groupby(["hora_utc", "categoria_altitud"], observed=True)["bias_orig"]
        .mean()
        .unstack("categoria_altitud")
        .reindex(columns=CATS)
        .reset_index()
        .rename(columns={"hora_utc": "hora"})
    )
    return result
 
 
@st.cache_data(show_spinner=False)
def compute_shap_global(path_shap: str, path_modelo: str, tipo="xgboost") -> pd.DataFrame:
    shap_vals = cargar_shap(path_shap)
    modelo    = cargar_modelo(path_modelo, tipo=tipo)
    
    # Extraemos los nombres de las features dependiendo de la librería
    if tipo == "xgboost":
        feature_names = modelo.get_booster().feature_names
    elif tipo == "catboost":
        feature_names = modelo.feature_names_
    else:
        feature_names = []


    # Si la matriz de SHAP tiene más columnas que variables, la recortamos.
    if shap_vals.shape[1] > len(feature_names):
        shap_vals = shap_vals[:, :len(feature_names)]

    shap_mean = np.abs(shap_vals).mean(axis=0)
    
    return (
        pd.DataFrame({"feature": feature_names, "shap_mean": shap_mean})
        .sort_values("shap_mean")
        .tail(15)
        .reset_index(drop=True)
    )
 
@st.cache_data(show_spinner=False)
def compute_shap_by_altitude(path_shap: str, path_modelo: str, path_df: str, tipo="xgboost") -> pd.DataFrame:
    shap_vals = cargar_shap(path_shap)
    modelo    = cargar_modelo(path_modelo, tipo=tipo)
    
    # Extraemos los nombres de las features dependiendo de la librería
    if tipo == "xgboost":
        feature_names = modelo.get_booster().feature_names
    elif tipo == "catboost":
        feature_names = modelo.feature_names_
    else:
        feature_names = []

    # 🛠️ Aplicamos la misma limpieza de seguridad por si acaso
    if shap_vals.shape[1] > len(feature_names):
        shap_vals = shap_vals[:, :len(feature_names)]

    df = cargar_df(path_df)
 
    FEAT = "elevation"
    if FEAT not in feature_names:
        st.warning(f"Feature '{FEAT}' no encontrada en el modelo.")
        return pd.DataFrame({"categoria_altitud": CATS, "shap_altitud": [0.0] * len(CATS)})
 
    df = df.copy()
    df["_shap_alt"] = shap_vals[:, feature_names.index(FEAT)]
    
    return (
        df.groupby("categoria_altitud", observed=True)["_shap_alt"]
        .mean()
        .reindex(CATS)
        .reset_index()
        .rename(columns={"_shap_alt": "shap_altitud"})
    )
    
@st.cache_data(show_spinner=False)
def get_global_mae_comparison(path_xgb: str, path_cat: str) -> pd.DataFrame:
    df_xgb = cargar_df(path_xgb)
    df_cat = cargar_df(path_cat)
    
    # Extraemos los MAE globales
    data = {
        "Modelo": ["Open-Meteo", 'CatBoost', "XGBoost"],
        "MAE": [
            df_xgb["error_orig_abs"].mean(),
            df_cat["error_corr_abs"].mean(),
            df_xgb["error_corr_abs"].mean()
        ]
    }
    return pd.DataFrame(data)
    
# ============== DF MAPA ===================
@st.cache_data(show_spinner=False)
def compute_map_data(path: str) -> pd.DataFrame:
    df = cargar_df(path)
    
    # Agrupamos por estación
    df_mapa = df.groupby(["station_id", "lat", "lon", "elevation", "categoria_altitud"], observed=True).agg(
        mae_orig=("error_orig_abs", "mean"),
        mae_corr=("error_corr_abs", "mean")
    ).reset_index()
    
    # Cálculos de mejora
    df_mapa["mejora_grados"] = df_mapa["mae_orig"] - df_mapa["mae_corr"]
    df_mapa["mejora_pct"] = (df_mapa["mejora_grados"] / df_mapa["mae_orig"]) * 100
    
    return df_mapa
    
# ================= GRAFICO DIURNO ====================
@st.cache_data(show_spinner="Calculando ciclo diurno...")
def compute_diurnal_bias(path_df: str) -> pd.DataFrame:
    df = cargar_df(path_df)
    
    # 3. Agrupamos por hora y categoría
    result = df.groupby(["hora_utc", "categoria_altitud"], observed=True).agg(
        bias_real=("bias_orig", "mean"),
        bias_corregido=("bias_corr", "mean")
    ).reset_index()
    
    return result

