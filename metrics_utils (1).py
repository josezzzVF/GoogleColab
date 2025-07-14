import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
import tensorflow as tf

# Clases multiclas
CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# Rutas a los modelos entrenados
MODEL_PATHS = {
    'CNN 2D Personalizada': '/content/models/cnn_2d_personalizada.keras',
    'DenseNet121':           '/content/models/densenet121.keras',
    'ResNet50':              '/content/models/resnet50.keras'
}

def matthews_corrcoef_multiclass(cm):
    mccs = []
    n = cm.shape[0]
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mccs.append((tp*tn - fp*fn)/denom if denom != 0 else 0)
    return float(np.mean(mccs))

def mcnemar_test(y_true, y_pred1, y_pred2):
    table = np.zeros((2,2), int)
    for t, p1, p2 in zip(y_true, y_pred1, y_pred2):
        if p1==t and p2!=t: table[0,1]+=1
        if p1!=t and p2==t: table[1,0]+=1
    b, c = table[0,1], table[1,0]
    if b+c==0: return 0.0, 1.0
    stat = ((abs(b-c)-1)**2)/(b+c) if (b+c)>25 else ((b-c)**2)/(b+c)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)

def evaluate_on_dataset_multiclass(test_dir):
    # 1) Cargar datos
    X, y_true = [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(test_dir, cls)
        for fn in os.listdir(folder):
            if fn.endswith('.npy'):
                arr = np.load(os.path.join(folder, fn)).astype(np.float32)
                if arr.ndim==2: arr=arr[...,None]
                X.append(arr); y_true.append(idx)
    X = np.stack(X); y_true = np.array(y_true)

    metrics_list = []
    confusion_paths = {}
    preds_per_model = []

    # 2) Iterar modelos
    for name, mp in MODEL_PATHS.items():
        model = load_model(mp, custom_objects={'auc': AUC()})

        # ajustar canales
        X_in = X.copy()
        if model.input_shape[-1]==1 and X_in.shape[-1]==3:
            X_in = tf.image.rgb_to_grayscale(X_in).numpy()
        if model.input_shape[-1]==3 and X_in.shape[-1]==1:
            X_in = np.repeat(X_in,3,axis=-1)

        preds = model.predict(X_in, batch_size=32)
        y_pred = np.argmax(preds, axis=1)
        preds_per_model.append(y_pred)

        rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
        cm  = confusion_matrix(y_true, y_pred)

        # guardar matriz
        import matplotlib.pyplot as plt, seaborn as sns
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
        plt.title(name); plt.tight_layout()
        fp = f'confusion_{name.replace(" ","_")}.png'
        plt.savefig(fp); plt.close()
        confusion_paths[name]=fp

        mcc   = matthews_corrcoef_multiclass(cm)
        kappa = cohen_kappa_score(y_true, y_pred)
        cmets = {f'{cls}_precision': rep[cls]['precision'] for cls in CLASSES}
        cmets.update({f'{cls}_recall':    rep[cls]['recall']    for cls in CLASSES})
        cmets.update({f'{cls}_f1':        rep[cls]['f1-score']   for cls in CLASSES})

        metrics_list.append({
            'model':    name,
            'accuracy': rep['accuracy'],
            'mcc':      mcc,
            'kappa':    kappa,
            **cmets
        })

        K.clear_session()

    # 3) McNemar en pares
    comparisons = {}
    n = len(preds_per_model)
    for i in range(n):
        for j in range(i+1,n):
            key = f'{i}_{j}'
            comparisons[key] = mcnemar_test(y_true, preds_per_model[i], preds_per_model[j])

    # 4) CSV resumen
    pd.DataFrame(metrics_list).to_csv('metrics_multiclass_comparison.csv', index=False)
    return metrics_list, comparisons, confusion_paths
