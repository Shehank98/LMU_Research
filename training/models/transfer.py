"""
Transfer-learning model builders for InceptionV3 and Xception.

Each architecture has two variants:
  _standalone  – full-resolution (256×256) direct classifier
  _extractor   – compact (128×128) model whose Dense(256) layer feeds classical ML

Explicit name= parameters on every Dense layer guarantee the names that
webapp/model_loader.py expects in FEATURE_LAYERS:
  InceptionV3 extractor → 'dense_2'
  Xception    extractor → 'dense_3'
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.models import Model

from config import N_CLASSES, FEATURE_DIM


# ── InceptionV3 ───────────────────────────────────────────────────────────────

def build_inception_standalone(input_size: int = 256) -> tf.keras.Model:
    """Frozen InceptionV3 → GlobalAvgPool → Dense(4, softmax)."""
    base = InceptionV3(input_shape=(input_size, input_size, 3),
                       weights='imagenet', include_top=False)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='dense')(x)

    model = Model(inputs=base.input, outputs=out, name='inception_standalone')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_inception_extractor(input_size: int = 128) -> tf.keras.Model:
    """
    Frozen InceptionV3 → Flatten → Dense(256, relu, name='dense_2')
    → Dropout(0.5) → Dense(4, softmax).
    Feature layer name 'dense_2' matches model_loader.py FEATURE_LAYERS.
    """
    base = InceptionV3(input_shape=(input_size, input_size, 3),
                       weights='imagenet', include_top=False)
    base.trainable = False

    x   = layers.Flatten()(base.output)
    x   = layers.Dense(FEATURE_DIM, activation='relu', name='dense_2')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out, name='inception_extractor')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ── Xception ──────────────────────────────────────────────────────────────────

def build_xception_standalone(input_size: int = 256) -> tf.keras.Model:
    """Frozen Xception → GlobalAvgPool → Dense(4, softmax)."""
    base = Xception(input_shape=(input_size, input_size, 3),
                    weights='imagenet', include_top=False)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='dense')(x)

    model = Model(inputs=base.input, outputs=out, name='xception_standalone')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_xception_extractor(input_size: int = 128) -> tf.keras.Model:
    """
    Frozen Xception → Flatten → Dense(256, relu, name='dense_3')
    → Dropout(0.5) → Dense(4, softmax).
    Feature layer name 'dense_3' matches model_loader.py FEATURE_LAYERS.
    """
    base = Xception(input_shape=(input_size, input_size, 3),
                    weights='imagenet', include_top=False)
    base.trainable = False

    x   = layers.Flatten()(base.output)
    x   = layers.Dense(FEATURE_DIM, activation='relu', name='dense_3')(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out, name='xception_extractor')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
