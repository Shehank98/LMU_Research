"""
Custom CNN builder.

One function covers both use cases:
  - Standalone classifier (256×256, feat_layer_name='dense')
  - Feature extractor     (128×128, feat_layer_name='dense')

The explicit name= parameter guarantees the layer name regardless of how many
Dense layers exist elsewhere in the Keras session — webapp/model_loader.py depends on it.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from config import N_CLASSES


def build_cnn(input_size: int, feature_dim: int = 256,
              feat_layer_name: str = 'dense') -> tf.keras.Model:
    """
    Architecture:
        4 × (Conv2D → MaxPool2D)
        Flatten
        Dense(feature_dim, relu, name=feat_layer_name)  ← feature extraction point
        Dropout(0.5)
        Dense(N_CLASSES, softmax)

    Compiled with Adam + SparseCategoricalCrossentropy.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(input_size, input_size, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(feature_dim, activation='relu', name=feat_layer_name),
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation='softmax'),
    ], name=f'cnn_{input_size}px')

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return model
