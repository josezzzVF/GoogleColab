import os
import glob
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from preprocessing import load_and_preprocess_image
from model_utils import predict_image_multiclass, CLASSES
from metrics_utils import evaluate_on_dataset_multiclass
from report_utils_sugarcane import (
    generate_pdf_report_multiclass,
    generate_comparison_report_multiclass
)

st.set_page_config(
    page_title="Diagnóstico Multiclase Caña de Azúcar",
    layout="wide"
)

st.title("🌾 Diagnóstico Enfermedades en Hojas de Caña de Azúcar")

MODEL_PATHS = {
    "CNN 2D Personalizada": "/content/models/cnn_2d_personalizada.keras",
    "DenseNet121":           "/content/models/densenet121.keras",
    "ResNet50":              "/content/models/resnet50.keras"
}

@st.cache_resource
def load_model_cached(name):
    return tf.keras.models.load_model(MODEL_PATHS[name])

# Selección de modelo para diagnóstico individual
model_choice = st.sidebar.selectbox("Modelo (diagnóstico individual)", list(MODEL_PATHS.keys()))
model = load_model_cached(model_choice)

DATASET_PATH = "/content/processed_sugarcane_leaves_npy"

# --- Diagnóstico individual ---
st.header("🔍 Diagnóstico Individual")

inp_mode = st.radio("Entrada", ["Imagen (PNG/JPG)", "Ejemplo preprocesado (.npy)"])
file = None

if inp_mode == "Imagen (PNG/JPG)":
    file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
else:
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    cls = st.selectbox("Clase de ejemplo", classes)
    files = glob.glob(os.path.join(DATASET_PATH, cls, "*.npy"))
    if files:
        sel = st.selectbox("Archivo .npy", files)
        file = sel

if file:
    # Carga y preprocesado
    if hasattr(file, "read"):
        arr, disp = load_and_preprocess_image(file)
    else:
        arr = load_and_preprocess_image_npy(file)
        disp = (arr.squeeze() * 255).astype("uint8")

    st.image(disp, caption="Entrada", use_column_width=True)

    idx, confs, heat = predict_image_multiclass(model, arr)
    st.markdown(f"**Predicción:** {CLASSES[idx]}  \n**Confianza:** {confs[idx]:.1%}")
    st.image(heat, caption="Heatmap", width=300)

    if st.button("📄 Descargar Reporte Individual"):
        pdfp = generate_pdf_report_multiclass(disp, heat, CLASSES[idx], confs, model_choice)
        with open(pdfp, "rb") as f:
            st.download_button(
                "Descargar PDF Individual",
                data=f,
                file_name=os.path.basename(pdfp),
                mime="application/pdf"
            )

# --- Comparación global ---
st.header("📊 Comparación de Modelos")

# Botón que genera todo y guarda la ruta en session_state
if st.button("Evaluar y Generar Reporte"):
    with st.spinner("Calculando métricas…"):
        metrics_list, comparisons, confusion_paths = evaluate_on_dataset_multiclass(DATASET_PATH)
        # guardo en estado para después poder descargar
        rpt_path = generate_comparison_report_multiclass(
            metrics_list=metrics_list,
            model_names=list(MODEL_PATHS.keys()),
            comparisons=comparisons,
            confusion_paths=confusion_paths
        )
        st.session_state['last_report'] = rpt_path

    # muestro resultados en tablas
    df_mets = pd.DataFrame(metrics_list).set_index("model")
    fmt = {"accuracy":"{:.2%}", "mcc":"{:.3f}", "kappa":"{:.3f}"}
    for c in CLASSES:
        fmt[f"{c}_precision"] = "{:.1%}"
        fmt[f"{c}_recall"]    = "{:.1%}"
        fmt[f"{c}_f1"]        = "{:.1%}"
    st.subheader("Métricas de Rendimiento")
    st.dataframe(df_mets.style.format(fmt))

    rows = []
    for key,(stat,p) in comparisons.items():
        i,j = map(int,key.split("_"))
        name1 = list(MODEL_PATHS.keys())[i]
        name2 = list(MODEL_PATHS.keys())[j]
        conclusion = (
            "Significativa (p < 0.05)"
            if p < 0.05 else
            "No significativa (p ≥ 0.05)"
        )
        rows.append({
            "Comparación": f"{name1} vs {name2}",
            "Chi²": stat,
            "p-value": p,
            "Conclusión": conclusion
        })
    df_mcn = pd.DataFrame(rows).set_index("Comparación")
    st.subheader("Pruebas de McNemar")
    st.dataframe(
        df_mcn.style.format({"Chi²":"{:.4f}", "p-value":"{:.6f}"})
    )

# Si ya hay un reporte generado, muestro el botón de descarga
if 'last_report' in st.session_state:
    st.download_button(
        "📥 Descargar Reporte PDF Comparativo",
        data=open(st.session_state['last_report'], 'rb'),
        file_name=os.path.basename(st.session_state['last_report']),
        mime="application/pdf"
    )
    st.success("✅ Reporte listo para descargar.")
