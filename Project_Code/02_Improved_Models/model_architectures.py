"""Model definitions. Every network takes RGB images scaled to [0, 1]."""

import tensorflow as tf
from tensorflow.keras import layers, Model

from settings import IMG_SIZE, CLASSES

N_CLASSES = len(CLASSES)


def augmentation():
    """Light augmentation, used only while training."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )


def build_cnn(size=IMG_SIZE):
    """Custom CNN: four conv blocks with batch normalisation and a global-average-pooling head.

    Global pooling (instead of Flatten) stops the classifier from tying decisions to
    fixed pixel positions such as the image border.
    """
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Activation("linear", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", name="features")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return Model(inp, out, name="cnn")


def build_transfer(kind, size=IMG_SIZE):
    """InceptionV3 or Xception with ImageNet weights and a new classification head.

    The backbone starts frozen; unfreeze_top() later opens its upper blocks for fine-tuning.
    """
    app = {"xception": tf.keras.applications.Xception, "inceptionv3": tf.keras.applications.InceptionV3}[kind]
    base = app(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    x = layers.Rescaling(2.0, offset=-1.0)(x)  # [0, 1] -> [-1, 1], as both backbones expect
    x = base(x, training=False)  # keep batch-norm statistics fixed
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", name="features")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return Model(inp, out, name=kind)


def unfreeze_top(model, first_layer):
    """Make the backbone trainable from first_layer onwards (batch-norm layers stay frozen)."""
    base = next(l for l in model.layers if isinstance(l, Model) and l.name != "augment")
    base.trainable = True
    trainable = False
    for layer in base.layers:
        if layer.name == first_layer:
            trainable = True
        layer.trainable = trainable and not isinstance(layer, layers.BatchNormalization)
