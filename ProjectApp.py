"""
Helmet Detection Web App
Upload your trained YOLO weights (best.pt) and an image -> get helmet / no-helmet counts.

Run:
    pip install ultralytics gradio opencv-python
    python app.py
Then open the link shown in the terminal (http://127.0.0.1:7860).
"""
import os
import sys

import cv2
import gradio as gr
from ultralytics import YOLO

DEFAULT_WEIGHTS = "best_v1.pt"   # used automatically if this file sits next to app.py
GREEN, RED = (0, 200, 0), (255, 0, 0)   # RGB, because Gradio images are RGB

_model_cache = {}


def load_model(path):
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def parse_names(text):
    return [n.strip().lower() for n in text.split(",") if n.strip()] if text else []


def classify(name, helmet_names, no_helmet_names):
    """Return 'helmet', 'no_helmet' or None for a class name."""
    n = name.lower().strip()
    if helmet_names or no_helmet_names:          # manual override
        if n in no_helmet_names:
            return "no_helmet"
        if n in helmet_names:
            return "helmet"
        return None
    words = set(n.replace("_", " ").replace("-", " ").split())
    if words & {"no", "without", "nohelmet"} or n.startswith("nohelmet"):
        return "no_helmet"
    if "helmet" in n:
        return "helmet"
    return None


def draw_box(img, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if color == RED else 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(y1 - h - 8, 0)), (x1 + w + 4, max(y1, h + 8)), color, -1)
    cv2.putText(img, label, (x1 + 2, max(y1 - 5, h + 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def detect(image, weights_file, conf, helmet_text, no_helmet_text):
    if image is None:
        raise gr.Error("Please upload an image.")

    weights = weights_file if weights_file else (DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else None)
    if weights is None:
        raise gr.Error("Please upload your trained weights file (best.pt).")

    model = load_model(weights)
    helmet_names, no_helmet_names = parse_names(helmet_text), parse_names(no_helmet_text)

    result = model.predict(image, conf=conf, verbose=False)[0]
    annotated = image.copy()
    helmets = no_helmets = 0

    for box, cls, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        name = model.names[int(cls)]
        kind = classify(name, helmet_names, no_helmet_names)
        if kind == "helmet":
            helmets += 1
            draw_box(annotated, box, f"Helmet {score:.2f}", GREEN)
        elif kind == "no_helmet":
            no_helmets += 1
            draw_box(annotated, box, f"NO HELMET {score:.2f}", RED)

    summary = (
        f"### Results\n"
        f"- 🟢 **With helmet:** {helmets}\n"
        f"- 🔴 **Without helmet:** {no_helmets}\n"
        f"- Total people detected: {helmets + no_helmets}\n\n"
        f"_Model classes: {', '.join(model.names.values())}_"
    )
    if helmets + no_helmets == 0:
        summary += ("\n\n⚠️ Nothing matched. Try lowering the confidence, or set the class names "
                    "under *Advanced* to match the model classes listed above.")
    return annotated, summary


with gr.Blocks(title="Helmet Detection") as demo:
    gr.Markdown("# 🪖 Helmet Detection\nUpload your trained YOLO model and an image to count riders with and without helmets.")
    with gr.Row():
        with gr.Column():
            weights_in = gr.File(label="Trained model (best.pt)", file_types=[".pt"], type="filepath")
            image_in = gr.Image(label="Input image", type="numpy")
            conf_in = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Confidence threshold")
            with gr.Accordion("Advanced: class names (only if auto-detection fails)", open=False):
                helmet_in = gr.Textbox(label="Class name(s) meaning 'helmet'", placeholder="e.g. With Helmet")
                no_helmet_in = gr.Textbox(label="Class name(s) meaning 'no helmet'", placeholder="e.g. Without Helmet")
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="Detections")
            summary_out = gr.Markdown()

    run_btn.click(detect, [image_in, weights_in, conf_in, helmet_in, no_helmet_in],
                  [image_out, summary_out])

if __name__ == "__main__":
    # In Colab a public link is needed; locally the normal address is enough.
    demo.launch(share="google.colab" in sys.modules)
