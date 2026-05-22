"""
Shared dataset loading utilities used by all training scripts.

Two loaders:
  get_tf_dataset()      – tf.data pipeline for training Keras models
  load_dataset_numpy()  – numpy arrays for sklearn feature extraction
"""
import os
import glob
import numpy as np
import cv2
import tensorflow as tf

from config import DATA_DIR, CLASS_NAMES, N_CLASSES


def get_tf_dataset(split: str, size: int, batch_size: int,
                   validation_split: float = 0.1, seed: int = 42):
    """
    Return (train_ds, val_ds) for split='training' or (test_ds, None) for split='testing'.

    Images are rescaled to [0, 1] inside the pipeline.
    Class order: alphabetical → glioma=0, meningioma=1, notumor=2, pituitary=3.
    """
    directory = os.path.join(DATA_DIR, split.capitalize())

    if split.lower() == 'testing':
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(size, size),
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds, None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset='training',
        image_size=(size, size),
        batch_size=batch_size,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset='validation',
        image_size=(size, size),
        batch_size=batch_size,
        seed=seed,
    )
    rescale = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    train_ds = train_ds.map(rescale, num_parallel_calls=tf.data.AUTOTUNE) \
                       .cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(rescale, num_parallel_calls=tf.data.AUTOTUNE) \
                     .cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def load_dataset_numpy(split: str, size: int):
    """
    Load all images into float32 numpy arrays (single normalisation: / 255.0).

    Returns:
        X  – (N, size, size, 3) float32
        y  – (N,) int32 class indices, same ordering as CLASS_NAMES
    """
    label_to_id = {cls: i for i, cls in enumerate(sorted(CLASS_NAMES))}
    directory   = os.path.join(DATA_DIR, split.capitalize())

    images, labels = [], []
    for cls_dir in sorted(glob.glob(os.path.join(directory, '*'))):
        cls_name = os.path.basename(cls_dir)
        if cls_name not in label_to_id:
            continue
        cls_id = label_to_id[cls_name]
        for img_path in glob.glob(os.path.join(cls_dir, '*')):
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (size, size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(cls_id)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    print(f'[data] {split}: {len(X)} images, size {size}×{size}')
    return X, y
