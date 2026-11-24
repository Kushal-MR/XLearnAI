"""
gradcam.py - Grad-CAM attention heatmaps for X-Learn AI.

Works with the VGG-style CNN in fracture_model.h5
(Conv/BN/ReLU/MaxPool x3 -> GlobalAveragePooling2D -> Dense -> Sigmoid).
No extra dependencies: only tensorflow, numpy and Pillow.

Usage:
    from gradcam import explain, overlay_heatmap
    prob, cam = explain(model, pil_image)
    heat_img = overlay_heatmap(pil_image, cam, strength=0.8)
"""
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)  # must match the size used in training

# Your sigmoid output is the probability of the class with index 1.
# The paper says high probability = NON-fractured (fractured = 0, alphabetical
# order from the dataset folders). If your app treats it the other way round,
# pass fracture_is_low=False.
FRACTURE_IS_LOW = True


def preprocess(pil_img, size=IMG_SIZE):
    """Resize + normalise exactly as in training (edit if you normalised differently)."""
    img = pil_img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def _feature_layer(model):
    """Last feature map before global pooling. If a pooling layer sits right before
    GlobalAveragePooling2D, step back one layer to get a sharper (32x32) map."""
    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    for i, layer in enumerate(layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) and i > 0:
            j = i - 1
            if isinstance(layers[j], (tf.keras.layers.MaxPooling2D,
                                      tf.keras.layers.AveragePooling2D)) and j > 0:
                j -= 1
            return layers[j]
    for layer in reversed(layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional feature layer found in the model.")


def make_gradcam(model, batch):
    """Grad-CAM for the class the AI actually predicted.
    Returns (cam in [0,1] at feature-map resolution, raw sigmoid output)."""
    feat_layer = _feature_layer(model)
    x = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        feats = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer is feat_layer:
                feats = x
                tape.watch(feats)
        prob = x[:, 0]
        score = tf.where(prob > 0.5, prob, 1.0 - prob)   # winning-class score
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2))                       # (1, C)
    cam = tf.reduce_sum(feats * weights[:, None, None, :], axis=-1)[0]  # (H, W)
    cam = tf.nn.relu(cam).numpy()
    # robust normalisation: ignore the top 1% outliers so the map isn't washed out
    cam = np.clip(cam / (np.percentile(cam, 99) + 1e-8), 0.0, 1.0)
    return cam, float(prob[0])


def _jet(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(pil_img, cam, strength=0.8, threshold=0.2):
    """Blend the heatmap onto the image. Areas below `threshold` stay untouched;
    above it the colour fades in, up to `strength` opacity (0-1)."""
    base = pil_img.convert("RGB")
    w, h = base.size
    resample = getattr(Image, "Resampling", Image).BICUBIC
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((w, h), resample)
    c = np.asarray(cam_img, dtype="float32") / 255.0
    heat = _jet(c) * 255.0
    a = (np.clip((c - threshold) / (1.0 - threshold), 0, 1) * strength)[..., None]
    out = np.asarray(base, dtype="float32") * (1 - a) + heat * a
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


def explain(model, pil_img):
    """One forward/backward pass. Returns (sigmoid_output, cam).
    Draw it with overlay_heatmap(pil_img, cam, strength)."""
    return_cam, prob = make_gradcam(model, preprocess(pil_img))
    return prob, return_cam


def fracture_probability(prob, fracture_is_low=FRACTURE_IS_LOW):
    """Convert the raw sigmoid output to P(fractured) for display."""
    return 1.0 - prob if fracture_is_low else prob
