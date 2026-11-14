"""Explainability methods used in Chapter 5. All maps are returned as (B, 256, 256) arrays scaled to [0, 1]."""
import os, numpy as np, tensorflow as tf, cv2
from common import MODELS

# Target layer = last spatial (4-D) layer, the same choice as the EAA-IoU stage of the training notebook.
TARGET = {'CNN': 'conv2d_7', 'InceptionV3': 'mixed10', 'Xception': 'block14_sepconv2_act'}
FILES = {'CNN': 'cnn_model.h5', 'InceptionV3': 'inceptionv3_model.h5', 'Xception': 'xception_model.h5'}

def load(name):
    return tf.keras.models.load_model(os.path.join(MODELS, FILES[name]), compile=False)

class CamModel:
    """Wraps a classifier so we get (feature maps, pre-softmax logits) in one forward pass."""
    def __init__(self, model, layer):
        self.model, self.dense, self.layer = model, model.layers[-1], layer
        self.seq = isinstance(model, tf.keras.Sequential)
        if not self.seq:
            self.sub = tf.keras.Model(model.inputs, [model.get_layer(layer).output, self.dense.input])
    def _forward(self, X, tape):
        if not self.seq:
            return self.sub(X, training=False)
        h, A = X, None                                     # Sequential (custom CNN): run layer by layer
        for l in self.model.layers[:-1]:
            h = l(h, training=False)
            if l.name == self.layer: A = h; tape.watch(A)
        return A, h
    def maps_and_grads(self, X, cls):
        X = tf.convert_to_tensor(X, tf.float32); cls = tf.convert_to_tensor(np.asarray(cls), tf.int32)
        with tf.GradientTape() as tape:
            A, h = self._forward(X, tape)
            logits = tf.matmul(h, self.dense.kernel) + self.dense.bias   # explain logits, not softmax
            s = tf.gather(logits, cls, axis=1, batch_dims=1)
        return A.numpy(), tape.gradient(s, A).numpy()

def _finish(cam, size=256):
    cam = np.maximum(cam, 0)
    mx = cam.reshape(len(cam), -1).max(1).reshape(-1, 1, 1); cam = np.divide(cam, mx, out=np.zeros_like(cam), where=mx > 0)
    return np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR) for c in cam])

def gradcam(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)
    w = G.mean(axis=(1, 2))                                   # channel weights = mean gradient
    return _finish(np.einsum('bhwc,bc->bhw', A, w))

def gradcam_pp(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)
    g2, g3 = G ** 2, G ** 3                                   # Chattopadhay et al. (2018), closed form
    denom = 2 * g2 + A.sum(axis=(1, 2), keepdims=True) * g3
    alpha = np.divide(g2, denom, out=np.zeros_like(g2), where=denom != 0)
    w = (alpha * np.maximum(G, 0)).sum(axis=(1, 2))
    return _finish(np.einsum('bhwc,bc->bhw', A, w))

def hirescam(cm, X, cls):
    A, G = cm.maps_and_grads(X, cls)                          # Draelos and Carin (2020): element-wise grad x act
    return _finish((A * G).sum(-1))

def occlusion(model, x, cls, patch=32, stride=16, batch=64):
    """Drop in target-class probability when a black square hides each region (Zeiler and Fergus, 2014)."""
    base = model.predict(x[None], verbose=0)[0, cls]
    pos = [(i, j) for i in range(0, 256 - patch + 1, stride) for j in range(0, 256 - patch + 1, stride)]
    heat = np.zeros((256, 256)); cnt = np.zeros((256, 256))
    for s in range(0, len(pos), batch):
        P = pos[s:s + batch]; xb = np.repeat(x[None], len(P), 0)
        for k, (i, j) in enumerate(P): xb[k, i:i + patch, j:j + patch] = 0
        p = model.predict(xb, verbose=0)[:, cls]
        for k, (i, j) in enumerate(P):
            heat[i:i + patch, j:j + patch] += base - p[k]; cnt[i:i + patch, j:j + patch] += 1
    heat = np.maximum(heat / np.maximum(cnt, 1), 0)
    return heat / heat.max() if heat.max() > 0 else heat

def lime_map(model, x, cls, num_samples=600, seed=42):
    from lime import lime_image
    from skimage.segmentation import quickshift
    exp = lime_image.LimeImageExplainer(random_state=seed)
    seg = lambda im: quickshift(im, kernel_size=4, max_dist=200, ratio=0.2, rng=seed)
    e = exp.explain_instance(x.astype('double'), lambda b: model.predict(b, batch_size=64, verbose=0),
                             labels=(cls,), top_labels=None, hide_color=0, num_samples=num_samples,
                             segmentation_fn=seg, random_seed=seed)
    w = dict(e.local_exp[cls]); m = np.vectorize(lambda s: w.get(s, 0.0))(e.segments).astype('float32')
    m = np.maximum(m, 0); return m / m.max() if m.max() > 0 else m

def brain_mask(x):
    g = (x.mean(-1) * 255).astype('uint8'); _, b = cv2.threshold(g, 20, 1, cv2.THRESH_BINARY)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n > 1: b = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype('uint8')
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full = np.zeros_like(b); cv2.drawContours(full, cnts, -1, 1, -1)
    return full.astype(bool)

def deletion_insertion(model, X, maps, cls, steps=20, batch=64):
    """Petsiuk et al. (2018). Deletion: remove most salient pixels first (set to 0); insertion: reveal them on a blurred copy."""
    n, H = len(X), X.shape[1] * X.shape[2]
    order = np.argsort(-maps.reshape(n, -1), axis=1)
    blur = np.stack([cv2.GaussianBlur(x, (51, 51), 25) for x in X])
    curves = {'deletion': np.zeros((n, steps + 1)), 'insertion': np.zeros((n, steps + 1))}
    for kind in curves:
        imgs = []
        for t in range(steps + 1):
            k = int(round(H * t / steps)); 
            for i in range(n):
                idx = order[i, :k]
                if kind == 'deletion':
                    im = X[i].reshape(-1, 3).copy(); im[idx] = 0
                else:
                    im = blur[i].reshape(-1, 3).copy(); im[idx] = X[i].reshape(-1, 3)[idx]
                imgs.append(im.reshape(X.shape[1:]))
        P = model.predict(np.asarray(imgs, 'float32'), batch_size=batch, verbose=0)
        P = P.reshape(steps + 1, n, -1)
        curves[kind] = P[:, np.arange(n), cls].T
    auc = {k: np.trapezoid(v, dx=1.0 / steps, axis=1) for k, v in curves.items()}
    return curves, auc
