import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121, ResNet50

# Clases para multiclasificación
CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

def load_models():
    models = {}

    # ---- CNN 2D Personalizada (1 canal) – con bloque extra ----
    inp1 = Input((128, 128, 1), name='input_grayscale')
    x = Conv2D(32, 3, activation='relu', padding='same')(inp1)
    x = MaxPooling2D()(x); x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x); x = BatchNormalization()(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x); x = BatchNormalization()(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x); x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out1 = Dense(len(CLASSES), activation='softmax')(x)
    m1 = Model(inputs=inp1, outputs=out1, name='CNN2D')
    m1.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    models['CNN 2D Personalizada'] = m1

    # ---- DenseNet121 (3 canales) ----
    inp2 = Input((128, 128, 3), name='input_rgb')
    base2 = DenseNet121(weights='imagenet', include_top=False, input_tensor=inp2)
    base2.trainable = False
    x2 = GlobalAveragePooling2D()(base2.output)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    out2 = Dense(len(CLASSES), activation='softmax')(x2)
    m2 = Model(inputs=inp2, outputs=out2, name='DenseNet121')
    m2.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    models['DenseNet121'] = m2

    # ---- ResNet50 (3 canales) con fine‑tuning ----
    inp3 = Input((128, 128, 3), name='input_rgb2')
    base3 = ResNet50(weights='imagenet', include_top=False, input_tensor=inp3)
    for layer in base3.layers:
        layer.trainable = False
    for layer in base3.layers[-10:]:
        layer.trainable = True
    x3 = GlobalAveragePooling2D()(base3.output)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.5)(x3)
    out3 = Dense(len(CLASSES), activation='softmax')(x3)
    m3 = Model(inputs=inp3, outputs=out3, name='ResNet50')
    m3.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    models['ResNet50'] = m3

    return models


def predict_image_multiclass(model, image_np):
    """
    image_np: array preprocesado (H,W) o (H,W,1) o (H,W,3)
    Retorna:
      - idx de la clase predicha
      - confidences (vector softmax)
      - heatmap Grad‑CAM
    """
    # Normalizar tipo float32
    img = image_np.astype(np.float32)
    # Expandir dims si es 2D
    if img.ndim == 2:
        img = img[..., np.newaxis]

    # Ajustar número de canales según el modelo
    in_ch = model.input_shape[-1]
    if in_ch == 1:
        # modelo grayscale
        if img.shape[-1] == 3:
            img = tf.image.rgb_to_grayscale(img).numpy()
    else:
        # modelo RGB
        if img.shape[-1] == 1:
            img = tf.image.grayscale_to_rgb(img).numpy()

    # Batch
    batch = np.expand_dims(img, 0)

    # Forward
    probs = model.predict(batch)[0]
    idx = int(np.argmax(probs))

    # Grad‑CAM
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    last_conv = convs[-1]
    grad_model = tf.keras.models.Model(model.inputs, [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch)
        loss = preds[:, idx]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    heatmap = tf.image.resize(cam[..., tf.newaxis], (img.shape[0], img.shape[1])).numpy().squeeze()

    return idx, probs, heatmap
