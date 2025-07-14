import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model_utils import load_models
from train_sugarcane import load_dataset, SugarcaneImageGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import json

# Número de clases para multiclasificación de hojas de caña\NUM_CLASSES = 5

def build_model(hp):
    model_type = hp.Choice('model_type', ['DenseNet121'])
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.6, step=0.1)
    dense_units = hp.Int('dense_units', 64, 512, step=64)

    models = load_models()
    base_model = models[model_type]

    x = base_model.output
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return model


def tune_hyperparameters(data_dir, max_epochs=50):
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective('val_auc', direction='max'),
        max_epochs=max_epochs,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='sugarcane_diagnosis',
        overwrite=True
    )

    train_cases, val_cases, train_labels, val_labels = load_dataset(data_dir)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_labels
    )
    class_weight = dict(enumerate(class_weights))

    class CustomTuner(kt.Hyperband):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model_type = hp.get('model_type')

            train_gen = SugarcaneImageGenerator(train_cases, train_labels, batch_size=8, augment=False, model_name=model_type)
            val_gen = SugarcaneImageGenerator(val_cases, val_labels, batch_size=8, augment=False, model_name=model_type)

            model = self.hypermodel.build(hp)
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=kwargs.get('epochs', 2),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
            return {'val_auc': max(history.history.get('val_auc', [0]))}

    tuner = CustomTuner(
        build_model,
        objective=kt.Objective('val_auc', direction='max'),
        max_epochs=max_epochs,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='sugarcane_diagnosis',
        overwrite=True
    )

    tuner.search()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nMejores hiperparámetros encontrados:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    model_type = best_hps.get('model_type')
    output_dir = os.path.join('hyperparameter_tuning', 'optimized', model_type)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_hps.values, f, indent=4)

    tuner.results_summary()
    return best_hps

if __name__ == '__main__':
    # Directorio base donde cada clase tiene su subcarpeta con .npy
    data_dir = '/content/processed_sugarcane_leaves_npy'
    os.makedirs('models', exist_ok=True)
    print('Iniciando búsqueda de hiperparámetros...')
    tune_hyperparameters(data_dir)
