import json
import os
import sys
from io import BytesIO
from pathlib import Path


def fail(message: str, code: int = 2) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
    raise SystemExit(code)


try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageStat
except Exception as exc:
    fail(f"Missing base image libraries: {exc}. Run setup-real-ai.cmd.")

try:
    import tensorflow as tf
except Exception as exc:
    fail(f"TensorFlow is not installed for this Python: {exc}. Run setup-real-ai.cmd, or use Docker/WSL if Windows TensorFlow install fails.")


def load_metadata(path: Path) -> dict:
    if path.exists():
      return json.loads(path.read_text(encoding="utf-8"))
    return {
        "backbone": "DenseNet121",
        "image_size": [224, 224],
        "threshold": 0.5,
        "positive_class": "PNEUMONIA",
        "gradcam_layer": "conv5_block16_2_conv",
        "metrics": {},
    }


def quality_check(image_path: Path) -> dict:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    stat = ImageStat.Stat(image)
    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    edge = float(np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32).std())
    aspect_ratio = width / max(1, height)
    checks = [
        {"label": "resolution", "passed": width >= 224 and height >= 224, "value": f"{width}x{height}"},
        {"label": "exposure", "passed": 35 <= brightness <= 220, "value": round(brightness, 2)},
        {"label": "contrast", "passed": contrast >= 25, "value": round(contrast, 2)},
        {"label": "sharpness", "passed": edge >= 8, "value": round(edge, 2)},
        {"label": "orientation", "passed": 0.45 <= aspect_ratio <= 1.65, "value": round(aspect_ratio, 2)},
    ]
    score = sum(1 for item in checks if item["passed"]) / len(checks)
    return {"score": round(score, 3), "passed": score >= 0.8, "checks": checks}


def preprocess(image_path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB").resize(size)
    raw = np.asarray(image).astype("float32")
    batch = np.expand_dims(raw.copy(), axis=0)
    batch = tf.keras.applications.densenet.preprocess_input(batch)
    return raw, batch


def get_backbone(model):
    conv_counts = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            count = sum(1 for child in layer.layers if isinstance(child, tf.keras.layers.Conv2D))
            if count:
                conv_counts.append((count, layer))
    if conv_counts:
        return sorted(conv_counts, key=lambda item: item[0])[-1][1]
    return model


def find_conv_layer(backbone, preferred_name: str | None):
    conv_layers = []
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
    if preferred_name:
        for layer in conv_layers:
            if layer.name == preferred_name:
                return layer
    if not conv_layers:
        raise RuntimeError("No Conv2D layer found for Grad-CAM.")
    return conv_layers[-1]


def build_gradcam_models(model, preferred_layer: str | None):
    backbone = get_backbone(model)
    last_conv_layer = find_conv_layer(backbone, preferred_layer)
    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output],
        name="local_gradcam_feature_extractor",
    )

    if backbone is model:
        return feature_extractor, None

    backbone_index = model.layers.index(backbone)
    classifier_input = tf.keras.Input(shape=backbone.output.shape[1:])
    x = classifier_input
    for layer in model.layers[backbone_index + 1:]:
        x = layer(x)
    classifier_head = tf.keras.Model(classifier_input, x, name="local_gradcam_classifier_head")
    return feature_extractor, classifier_head


def make_gradcam(model, raw: np.ndarray, batch: np.ndarray, metadata: dict, output_path: Path) -> str | None:
    try:
        feature_extractor, classifier_head = build_gradcam_models(model, metadata.get("gradcam_layer"))
        image_tensor = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            conv_outputs, features = feature_extractor(image_tensor, training=False)
            tape.watch(conv_outputs)
            predictions = classifier_head(features, training=False) if classifier_head else model(image_tensor, training=False)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap = heatmap / max(float(np.max(heatmap)), 1e-8)

        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((raw.shape[1], raw.shape[0]))
        heat = np.asarray(heatmap_img.convert("L"), dtype=np.float32) / 255.0
        red = np.zeros_like(raw)
        red[..., 0] = 255
        overlay = np.clip(raw * 0.68 + red * heat[..., None] * 0.48, 0, 255).astype("uint8")
        Image.fromarray(overlay).save(output_path)
        return None
    except Exception as exc:
        return str(exc)


def explain(probability: float, threshold: float) -> dict:
    predicted_class = "PNEUMONIA" if probability >= threshold else "NORMAL"
    uncertainty = max(0.0, 1.0 - abs(probability - threshold) / max(threshold, 1.0 - threshold, 1e-6))
    confidence = 1.0 - uncertainty
    if uncertainty >= 0.65:
        uncertainty_level = "high"
    elif uncertainty >= 0.35:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "low"
    return {
        "predictedClass": predicted_class,
        "confidenceScore": round(float(confidence), 4),
        "uncertaintyScore": round(float(uncertainty), 4),
        "uncertaintyLevel": uncertainty_level,
    }


def main() -> None:
    if len(sys.argv) != 3:
        fail("Usage: infer.py INPUT_IMAGE OUTPUT_DIR")
    image_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(os.environ.get("MODEL_PATH", ""))
    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", ""))
    if not model_path.exists():
        fail(f"Model file not found: {model_path}")

    metadata = load_metadata(metadata_path)
    threshold = float(metadata.get("threshold", 0.5))
    size = tuple(metadata.get("image_size", [224, 224]))

    quality = quality_check(image_path)
    model = tf.keras.models.load_model(model_path)
    raw, batch = preprocess(image_path, size)
    probability = float(model.predict(batch, verbose=0)[0][0])
    gradcam_error = make_gradcam(model, raw, batch, metadata, output_dir / "gradcam.png")
    details = explain(probability, threshold)

    result = {
        "ok": True,
        "modelVersion": f"{metadata.get('backbone', 'DenseNet121')}-local-real",
        "probabilityPneumonia": probability,
        "threshold": threshold,
        "quality": quality,
        "metrics": metadata.get("metrics", {}),
        "gradcamError": gradcam_error,
        **details,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

