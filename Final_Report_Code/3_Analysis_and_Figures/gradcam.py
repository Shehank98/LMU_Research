"""Grad-CAM for the three deep networks and the IoU used by EAA-IoU (same method as the training notebook).

Grad-CAM weights each feature map of the last convolutional layer by the mean gradient of the class score.
The custom CNN ends in Flatten + Dense, where averaging the gradients loses their spatial pattern, so for the
CNN the gradient is multiplied with the activation at every location before summing over the channels
(Draelos and Carin, 2020). For InceptionV3 and Xception (global average pooling) both forms give the same map.
"""

import os
import numpy as np
import cv2
import tensorflow as tf

from settings import MODELS

TARGET = {"CNN": "conv2d_7", "InceptionV3": "mixed10", "Xception": "block14_sepconv2_act"}
FILES = {"CNN": "cnn_model.h5", "InceptionV3": "inceptionv3_model.h5", "Xception": "xception_model.h5"}


def load(name):
    return tf.keras.models.load_model(os.path.join(MODELS, FILES[name]), compile=False)


class CamModel:
    """Returns (feature maps of the target layer, gradient of the class logit) for a batch of images."""

    def __init__(self, model, layer):
        self.model, self.layer, self.dense = model, layer, model.layers[-1]
        self.seq = isinstance(model, tf.keras.Sequential)
        if not self.seq:
            self.sub = tf.keras.Model(model.inputs, [model.get_layer(layer).output, self.dense.input])

    def maps_and_grads(self, X, cls):
        X = tf.convert_to_tensor(X, tf.float32)
        cls = tf.convert_to_tensor(np.asarray(cls), tf.int32)
        with tf.GradientTape() as tape:
            if self.seq:  # custom CNN: run the layers one by one and keep the target layer's output
                h, A = X, None
                for l in self.model.layers[:-1]:
                    h = l(h, training=False)
                    if l.name == self.layer:
                        A = h
                        tape.watch(A)
            else:
                A, h = self.sub(X, training=False)
            logits = tf.matmul(h, self.dense.kernel) + self.dense.bias  # explain the logit, not the softmax
            score = tf.gather(logits, cls, axis=1, batch_dims=1)
        return A.numpy(), tape.gradient(score, A).numpy()


def _finish(cam, size=256):
    cam = np.maximum(cam, 0)
    mx = cam.reshape(len(cam), -1).max(1).reshape(-1, 1, 1)
    cam = np.divide(cam, mx, out=np.zeros_like(cam), where=mx > 0)
    return np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR) for c in cam])


def gradcam(name, cam_model, X, cls):
    A, G = cam_model.maps_and_grads(X, cls)
    if name == "CNN":  # element-wise form for the Flatten + Dense head
        return _finish((A * G).sum(-1))
    w = G.mean(axis=(1, 2))  # Grad-CAM: one weight per feature map
    return _finish(np.einsum("bhwc,bc->bhw", A, w))


def iou(a, b, t=0.5):
    A, B = a >= t, b >= t
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union else 0.0
