import os
import numpy as np
from preprocessing import preprocess_image
from skimage.io import imread
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from model_utils import load_models
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def load_dataset(data_dir, test_size=0.2, random_state=42):
    classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    cases, labels = [], []
    for idx, label in enumerate(classes):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No se encontró la carpeta de clase: {class_dir}")
        for fname in os.listdir(class_dir):
            if fname.endswith('.npy'):
                cases.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return train_test_split(cases, labels, test_size=test_size,
                            random_state=random_state, stratify=labels)

class SugarcaneImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, cases, labels, batch_size=32, augment=False, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.cases = cases
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.model_name = model_name
        self.indices = np.arange(len(cases))
        if augment:
            self.augmenter = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='reflect'
            )
        else:
            self.augmenter = None

    def __len__(self):
        return int(np.ceil(len(self.cases) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_cases = [self.cases[i] for i in batch_idx]
        batch_labels = [self.labels[i] for i in batch_idx]

        images = []
        for path in batch_cases:
            img = np.load(path).astype(np.float32)
            # Si es escala de grises, expandir
            if img.ndim == 2:
                img = img[..., np.newaxis]
            # Convertir a RGB para modelos 3-canales
            if self.model_name in ("DenseNet121", "ResNet50"):
                if img.shape[-1] == 1:
                    img = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img)).numpy()
            # Para CNN 2D personalizada, asegurar 1 canal
            else:
                if img.shape[-1] == 3:
                    img = tf.image.rgb_to_grayscale(tf.convert_to_tensor(img)).numpy()
            images.append(img)

        x = np.stack(images, axis=0)
        # One-hot para 5 clases
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=5)

        if self.augmenter:
            x, y = next(self.augmenter.flow(x, y, batch_size=self.batch_size, shuffle=False))

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def train_model(model, train_gen, val_gen, epochs, model_name, class_weight=None):
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}.keras',
            monitor='val_auc', mode='max', save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks,
        class_weight=class_weight, verbose=1
    )
    pd.DataFrame(history.history).to_csv(f'models/{model_name}_history.csv', index=False)
    return history

if __name__ == "__main__":
    data_dir = "/content/processed_sugarcane_leaves_npy"
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_cases, val_cases, train_labels, val_labels = load_dataset(data_dir)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight = dict(enumerate(weights))

    models = load_models()
    for name, model in models.items():
        print(f"\n=== Entrenando {name} ===")
        train_gen = SugarcaneImageGenerator(
            train_cases, train_labels,
            batch_size=32, augment=True, model_name=name
        )
        val_gen = SugarcaneImageGenerator(
            val_cases, val_labels,
            batch_size=32, augment=False, model_name=name
        )
        train_model(
            model, train_gen, val_gen,
            epochs=100,
            model_name=name.lower().replace(" ", "_"),
            class_weight=class_weight
        )
