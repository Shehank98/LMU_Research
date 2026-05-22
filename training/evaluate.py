"""
Shared evaluation helpers used by every training script.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from config import CLASS_NAMES


def print_results(y_true, y_pred, title: str = ''):
    acc = accuracy_score(y_true, y_pred)
    if title:
        print(f'\n{"─" * 50}')
        print(f'  {title}')
        print(f'{"─" * 50}')
    print(f'  Accuracy: {acc * 100:.2f}%')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return acc


def plot_confusion_matrix(y_true, y_pred, title: str, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    if show:
        plt.tight_layout()
        plt.show()


def plot_training_history(history, title: str = ''):
    epochs = range(1, len(history.history['accuracy']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if title:
        fig.suptitle(title)

    axes[0].plot(epochs, history.history['accuracy'],     label='Train')
    axes[0].plot(epochs, history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, history.history['loss'],     label='Train')
    axes[1].plot(epochs, history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def extract_features(keras_model, layer_name: str,
                     X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """
    Return activations from a named layer for every row in X.
    Uses a sub-model so the original model is unchanged.
    """
    from tensorflow.keras.models import Model as KModel
    extractor = KModel(inputs=keras_model.input,
                       outputs=keras_model.get_layer(layer_name).output)
    return extractor.predict(X, batch_size=batch_size, verbose=1)
