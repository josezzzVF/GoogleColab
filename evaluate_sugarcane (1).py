import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import tensorflow as tf

# Clases de enfermedades en hojas de caña de azúcar
CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']


def load_test_dataset(test_dir):
    test_cases, test_labels = [], []
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(test_dir, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No se encontró la carpeta de clase: {class_dir}")
        for fname in os.listdir(class_dir):
            if fname.endswith('.npy'):
                test_cases.append(os.path.join(class_dir, fname))
                test_labels.append(idx)
    return test_cases, test_labels


def read_processed_image(path):
    """
    Carga un .npy y siempre devuelve un array con 3 canales (RGB).
    """
    image = np.load(path).astype(np.float32)
    # Si es (H,W), agrandar a (H,W,1)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    # Si es un solo canal, convertir a RGB
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image)).numpy()
    # Ahora image.shape es (H, W, 3)
    return image


def evaluate_models(test_dir, output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    figs_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)

    # Cargar casos y etiquetas
    test_cases, test_labels = load_test_dataset(test_dir)
    y_true = np.array(test_labels)
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))

    # Cargar modelos entrenados
    models = {
        'DenseNet121': load_model(
            '/content/models/densenet121_sugarcane.keras',
            custom_objects={'AUC': AUC, 'Precision': Precision, 'Recall': Recall}
        )
    }
    results = []

    for name, model in models.items():
        print(f"\nEvaluando modelo: {name}")
        X = []
        # Leer y convertir cada imagen a RGB
        for case in tqdm(test_cases, desc=f'Procesando imágenes para {name}'):
            img = read_processed_image(case)
            X.append(img)
        X = np.stack(X, axis=0)  # (N, H, W, 3)

        # Predicción softmax
        y_score = model.predict(X, batch_size=32)  # (N, num_classes)

        # Convertir scores a etiquetas one-hot
        y_pred_bin = np.zeros_like(y_score, dtype=int)
        idxs = np.argmax(y_score, axis=1)
        y_pred_bin[np.arange(len(idxs)), idxs] = 1
        y_pred_labels = idxs

        # Métricas de clasificación
        report = classification_report(
            y_true,
            y_pred_labels,
            target_names=CLASSES,
            output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred_labels)

        # ROC y PR por clase
        roc_auc = {}
        avg_prec = {}
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[cls] = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            avg_prec[cls] = average_precision_score(y_true_bin[:, i], y_score[:, i])

            # Guardar curvas ROC
            plt.figure(figsize=(6,4))
            plt.plot(fpr, tpr, label=f'AUC={roc_auc[cls]:.2f}')
            plt.plot([0,1], [0,1], 'k--')
            plt.title(f'ROC - {name} - {cls}')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.savefig(os.path.join(figs_dir, f'roc_{name.lower()}_{cls}.png'))
            plt.close()

            # Guardar curvas Precision-Recall
            plt.figure(figsize=(6,4))
            plt.plot(recall, precision, label=f'AP={avg_prec[cls]:.2f}')
            plt.title(f'Precision-Recall - {name} - {cls}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig(os.path.join(figs_dir, f'pr_{name.lower()}_{cls}.png'))
            plt.close()

        # Añadir resultados al resumen
        entry = {
            'Modelo': name,
            'Accuracy': report['accuracy']
        }
        entry.update({f'{cls}_Recall': report[cls]['recall'] for cls in CLASSES})
        entry.update({f'{cls}_ROC_AUC': roc_auc[cls] for cls in CLASSES})
        entry.update({f'{cls}_AP': avg_prec[cls] for cls in CLASSES})
        results.append(entry)

    # DataFrame y CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(df.to_markdown(index=False))
    return df


if __name__ == '__main__':
    test_dir = '/content/processed_sugarcane_leaves_npy'
    evaluate_models(test_dir)
