pip install -r requirements.txt

### Running
python -m streamlit run app.py

### is equivalent to:
streamlit run app.py

### Estructura

mi_proyecto/
│
├── data/
│   └── df_test.parquet
├── models/
│   ├── modelo_altura.json
│   └── shap_values.pkl
│
│
├── app.py                   <-- Solo maneja el menú lateral y la navegación.
│
└── pages/                    
    ├── 1_Resumen.py         
    └── 2_Mapa.py            

