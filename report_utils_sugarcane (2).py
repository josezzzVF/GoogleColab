from fpdf import FPDF
import time
import os

# Clases multiclasificación
CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

def generate_pdf_report_multiclass(
    image_display, heatmap, predicted_class, confidences, model_name,
    output_path='diagnostico_sugarcane.pdf'
):
    import matplotlib.pyplot as plt
    # 1) Combinar imagen + heatmap
    tmp = 'tmp_combined.png'
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image_display, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(image_display, cmap='gray')
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].axis('off')
    ax[1].set_title('Heatmap')
    fig.tight_layout()
    fig.savefig(tmp, dpi=300)
    plt.close(fig)

    # 2) Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Reporte Diagnóstico Multiclase', ln=1, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Modelo: {model_name}', ln=1)
    pdf.cell(0, 8, f'Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
    pdf.cell(0, 8, f'Predicción: {predicted_class}', ln=1)
    pdf.ln(2)
    pdf.cell(0, 8, 'Confidencias:', ln=1)
    for cls, c in zip(CLASSES, confidences):
        pdf.cell(0, 6, f'  • {cls}: {c*100:.2f}%', ln=1)
    pdf.ln(4)
    pdf.image(tmp, w=180)
    pdf.output(output_path)
    os.remove(tmp)
    return output_path

def generate_comparison_report_multiclass(
    metrics_list, model_names, comparisons, confusion_paths,
    output_path='comparacion_sugarcane.pdf'
):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(True, 15)

    # Portada
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Reporte Comparativo Multiclase', ln=1, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
    pdf.ln(5)

    # Tabla de métricas
    headers = ['Modelo', 'Accuracy', 'MCC', 'Kappa']
    for cls in CLASSES:
        headers += [f'{cls} P', f'{cls} R', f'{cls} F1']
    n_cols = len(headers)
    col_w = (pdf.w - 20) / n_cols  # margen 10mm izquierda/derecha

    pdf.set_font('Arial', 'B', 10)
    for h in headers:
        pdf.cell(col_w, 8, h, border=1, align='C')
    pdf.ln()

    pdf.set_font('Arial', '', 9)
    for m in metrics_list:
        pdf.cell(col_w, 8, m['model'], border=1)
        pdf.cell(col_w, 8, f"{m['accuracy']:.3f}", border=1, align='C')
        pdf.cell(col_w, 8, f"{m['mcc']:.3f}", border=1, align='C')
        pdf.cell(col_w, 8, f"{m['kappa']:.3f}", border=1, align='C')
        for cls in CLASSES:
            pdf.cell(col_w, 8, f"{m[f'{cls}_precision']:.3f}", border=1, align='C')
            pdf.cell(col_w, 8, f"{m[f'{cls}_recall']:.3f}", border=1, align='C')
            pdf.cell(col_w, 8, f"{m[f'{cls}_f1']:.3f}", border=1, align='C')
        pdf.ln()

    # Pruebas de McNemar
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Pruebas de McNemar', ln=1)
    pdf.set_font('Arial', '', 10)
    pdf.ln(2)

    # Encabezado tabla McNemar con conclusión
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, 'Comparación', border=1, align='C')
    pdf.cell(50, 8, 'Chi2', border=1, align='C')
    pdf.cell(50, 8, 'p-value', border=1, align='C')
    pdf.cell(130, 8, 'Conclusión', border=1, align='C')
    pdf.ln()
    pdf.set_font('Arial', '', 10)

    for key, (stat, p_val) in comparisons.items():
        i, j = map(int, key.split('_'))
        name1 = model_names[i]
        name2 = model_names[j]
        # p-value con 6 decimales
        p_text = f"{p_val:.6f}"
        # conclusión
        if p_val < 0.05:
            conclusion = (
                f"Existe una diferencia estadísticamente significativa en el rendimiento "
                f"de {name1} y {name2} (p < 0.05)."
            )
        else:
            conclusion = (
                f"No hay una diferencia estadísticamente significativa en el rendimiento "
                f"de {name1} y {name2} (p ≥ 0.05)."
            )
        pdf.cell(60, 8, f"{name1} vs {name2}", border=1)
        pdf.cell(50, 8, f"{stat:.4f}", border=1, align='C')
        pdf.cell(50, 8, p_text, border=1, align='C')
        pdf.cell(130, 8, conclusion, border=1)
        pdf.ln()

    # Matrices de confusión
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Matrices de Confusión', ln=1)
    pdf.ln(3)

    img_w = (pdf.w - 30) / 2
    x_positions = [10, 15 + img_w]
    y0 = pdf.get_y()

    for idx, name in enumerate(model_names):
        x = x_positions[idx % 2]
        if idx % 2 == 0 and idx > 0:
            y0 += img_w * 0.6 + 15
        pdf.set_xy(x, y0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(img_w, 6, name, ln=2, align='C')
        pdf.image(confusion_paths[name], x=x, y=pdf.get_y(), w=img_w)

    pdf.output(output_path)
    return output_path
