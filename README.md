🌱 **Sistema de Diagnóstico de Enfermedades en Hojas de Caña de Azúcar**

🧩 1. Introducción

Este proyecto corresponde a un Sistema de Diagnóstico Automático de Enfermedades en Hojas de Caña de Azúcar, desarrollado mediante técnicas de Deep Learning y análisis de imágenes RGB. El sistema emplea Redes Neuronales Convolucionales (CNN) entrenadas previamente para reconocer patrones visuales característicos de distintas enfermedades presentes en las hojas.

La aplicación ha sido diseñada como una herramienta de apoyo académico, investigativo y agrícola, permitiendo obtener diagnósticos rápidos, claros y fáciles de interpretar. Su objetivo principal es contribuir a la detección temprana de enfermedades, optimizando el tiempo de evaluación y apoyando la toma de decisiones en el manejo del cultivo.

🎯 2. Objetivo del sistema

Diagnosticar enfermedades en hojas de caña de azúcar a partir de imágenes.

Facilitar la evaluación automática mediante modelos entrenados.

Mostrar resultados claros y comprensibles para el usuario.

🧰 3. Requisitos del sistema
💻 3.1 Hardware

Computadora con al menos 8 GB de RAM (recomendado 16 GB).

Conexión a internet (para uso en Google Colab o Streamlit).

🧪 3.2 Software

Python 3.9 o superior.

Google Colab (recomendado) o entorno local.

Navegador web actualizado (Chrome, Edge o Firefox).

📦 3.3 Librerías principales

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Scikit-learn

Streamlit (para la aplicación interactiva)

📂 4. Estructura del proyecto

El proyecto se organiza de la siguiente manera:

dataset/ → Imágenes de hojas de caña de azúcar organizadas por clases.

models/ → Modelos entrenados (.keras o .h5).

notebooks/ → Notebooks de entrenamiento y evaluación.

app_sugarcane.py → Aplicación principal en Streamlit.

utils/ → Funciones auxiliares (preprocesamiento y predicción).

☁️ 5. Uso del sistema en Google Colab
5.1 Abrir el proyecto

Ingrese a Google Colab.

Cargue el notebook principal del proyecto.

Monte Google Drive si el dataset o modelos están almacenados allí.

5.2 Cargar el modelo entrenado

Ejecute la celda correspondiente para cargar el modelo previamente entrenado (CNN personalizada, DenseNet121 o ResNet50).

5.3 Evaluación del modelo

Ejecute las celdas de evaluación.

Revise métricas como precisión, matriz de confusión y reporte de clasificación.

🖥️ 6. Uso de la aplicación Streamlit
6.1 Ejecutar la aplicación

Desde la terminal o Colab, ejecute:

streamlit run app_sugarcane.py
6.2 Interfaz de usuario

La aplicación permite:

Cargar una imagen .jpg o .png de una hoja de caña de azúcar.

Visualizar la imagen cargada.

Ejecutar el diagnóstico con el modelo seleccionado.

6.3 Diagnóstico

Seleccione o cargue una imagen.

Presione el botón Diagnosticar.

El sistema mostrará:

La enfermedad detectada.

El nivel de confianza del modelo.

📊 7. Interpretación de resultados

Clase predicha: Enfermedad identificada en la hoja.

Probabilidad: Nivel de seguridad del modelo en la predicción.

Resultados con baja confianza deben ser verificados manualmente.

✅ 8. Buenas prácticas de uso

Utilizar imágenes claras y bien iluminadas.

Evitar fondos complejos.

Usar hojas completas y sin recortes excesivos.

⚠️ 9. Limitaciones del sistema

El diagnóstico depende de la calidad del dataset de entrenamiento.

No reemplaza la evaluación de un especialista agrícola.

Puede fallar ante enfermedades no incluidas en el entrenamiento.

🔄 10. Soporte y mantenimiento

Reentrenar el modelo al añadir nuevas clases.

Actualizar librerías periódicamente.

Validar el rendimiento con nuevos datos.

👨‍💻 11. Créditos

Proyecto desarrollado con fines académicos para el diagnóstico de enfermedades en hojas de caña de azúcar mediante técnicas de Deep Learning.
