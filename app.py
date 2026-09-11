"""Canvas Critic - Gradio UI.

Model logic lives in local_model.py / remote_model.py; routing and failover in
router.py; prompts and identifiers in config.py.
"""

import gradio as gr
from PIL import Image

from config import LOCAL_MODEL, REMOTE_MODEL
from router import score_artwork


def preview_upload(file_path: str | None):
    if not file_path:
        return None, None, "Upload an image to begin."

    try:
        with Image.open(file_path) as image:
            preview = image.convert("RGB")
        return preview, file_path, "Image ready for critique."
    except Exception as error:
        return None, None, f"Could not read that image: {error}"


def clear_workspace():
    return None, None, None, "Upload an image to begin.", "Your critique will appear here."


CSS = """
.gradio-container {
    width: min(1180px, 96%) !important;
    margin: 0 auto !important;
}
#app-title { text-align: center; margin-bottom: 0; }
#app-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin: 4px auto 22px;
}
.workspace-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.07);
}
#critique-output {
    min-height: 430px;
    padding: 20px;
    border-radius: 12px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
}
#status-line { color: var(--body-text-color-subdued); font-size: 0.9rem; }
#model-note { color: var(--body-text-color-subdued); font-size: 0.86rem; }
@media (max-width: 768px) {
    .workspace-panel { padding: 10px; }
    #critique-output { min-height: 260px; }
}
"""


with gr.Blocks(title="Canvas Critic") as demo:
    image_path_state = gr.State()

    with gr.Sidebar():
        gr.Markdown("### Hosted model access")
        gr.LoginButton()
        gr.Markdown(
            "Sign in with Hugging Face to use the hosted model on your own "
            "inference credits. The local model does not require a login.",
            elem_id="model-note",
        )

    gr.Markdown("# 🎨 Canvas Critic", elem_id="app-title")
    gr.Markdown(
        "Upload an artwork and receive an aesthetic score out of 100, plus "
        "concrete suggestions for improving it.",
        elem_id="app-subtitle",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="workspace-panel"):
            gr.Markdown("### 1. Add your artwork")
            upload_button = gr.UploadButton(
                "Upload an image",
                file_types=["image"],
                file_count="single",
                variant="primary",
            )
            image_preview = gr.Image(label="Artwork preview", interactive=False, height=360)
            upload_status = gr.Markdown("Upload an image to begin.", elem_id="status-line")

            with gr.Accordion("Scoring settings", open=False):
                max_tokens = gr.Slider(
                    minimum=512,
                    maximum=4096,
                    value=2048,
                    step=256,
                    label="Maximum response tokens",
                    info="The hosted model spends part of this budget on internal reasoning.",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.4,
                    step=0.1,
                    label="Temperature",
                    info="0 makes results reproducible.",
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                )
                use_local_model = gr.Checkbox(
                    label="Switch to local model",
                    value=False,
                    info=f"Runs {LOCAL_MODEL} in this Space instead of the hosted API.",
                )
                gr.Markdown(
                    f"Hosted: `{REMOTE_MODEL}` — a general vision LLM.  \n"
                    f"Local: `{LOCAL_MODEL}` — a model trained specifically to "
                    "score image aesthetics.",
                    elem_id="model-note",
                )

            with gr.Row():
                score_button = gr.Button("Score this artwork", variant="primary", scale=3)
                clear_button = gr.Button("Clear", scale=1)

        with gr.Column(scale=6, elem_classes="workspace-panel"):
            gr.Markdown("### 2. Critique")
            critique_output = gr.Markdown(
                "Your critique will appear here.",
                elem_id="critique-output",
            )
            model_status = gr.Markdown("", elem_id="status-line")

    upload_button.upload(
        fn=preview_upload,
        inputs=upload_button,
        outputs=[image_preview, image_path_state, upload_status],
    )
    score_button.click(
        fn=score_artwork,
        inputs=[
            image_path_state,
            max_tokens,
            temperature,
            top_p,
            use_local_model,
        ],
        outputs=[critique_output, model_status],
    )
    clear_button.click(
        fn=clear_workspace,
        outputs=[
            upload_button,
            image_preview,
            image_path_state,
            upload_status,
            critique_output,
        ],
    ).then(lambda: "", outputs=model_status)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
