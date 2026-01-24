import gradio as gr
from ..models.detector import SafetyHelmetDetector

def launch_app():
    detector = SafetyHelmetDetector()

    def predict_image(image):
        return detector.predict(image)

    with gr.Blocks() as iface:
        gr.Markdown("# ⛑️ Hard Hat Detection (TFLite Model)")
        gr.Markdown("Upload an image to detect helmets (green), heads without helmets (red), and people (blue).")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output = gr.Image(type="numpy", label="Detected Objects")

        btn = gr.Button("Detect Objects")
        btn.click(fn=predict_image, inputs=image_input, outputs=image_output)

    iface.launch()

if __name__ == "__main__":
    launch_app()
