import base64
import mimetypes
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import pipeline


REMOTE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LOCAL_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEFAULT_RUBRIC = """Evaluate the artwork as a thoughtful art critic. Consider:
- composition and visual hierarchy
- color, lighting, and tonal control
- technique and execution
- originality and emotional impact

Judge only the artwork visible in the image. Be constructive, specific, and respectful."""

local_pipe = None


def build_prompt(rubric: str) -> str:
    return f"""{rubric.strip() or DEFAULT_RUBRIC}

Return your assessment in exactly this general structure:

## Score: X/10
### First impression
One or two concise sentences.

### What works
- Two or three specific strengths grounded in visible details.

### What could improve
- Two or three specific, actionable suggestions.

### Verdict
One concise closing sentence.

The score must be a single number from 1 through 10. Do not claim to know the
artist's identity, intent, or process unless it is plainly visible in the image."""


def preview_upload(file_path: str | None):
    if not file_path:
        return None, None, "Upload an image to begin."

    try:
        with Image.open(file_path) as image:
            preview = image.convert("RGB")
        return preview, file_path, "Image ready for critique."
    except Exception as error:
        return None, None, f"Could not read that image: {error}"


def image_as_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def remote_score(image_path, prompt, max_tokens, temperature, top_p) -> str:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise gr.Error(
            "The hosted model needs an HF_TOKEN Space secret. Add it in the "
            "Space settings, or select 'Switch to local model'."
        )

    client = InferenceClient(provider="auto", api_key=hf_token)
    response = client.chat.completions.create(
        model=REMOTE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_as_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


@spaces.GPU
def local_score(image_path, prompt, max_tokens, temperature, top_p) -> str:
    global local_pipe

    if local_pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        local_pipe = pipeline(
            "image-text-to-text",
            model=LOCAL_MODEL,
            device=device,
            dtype=dtype,
        )

    with Image.open(image_path) as uploaded_image:
        image = uploaded_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    output = local_pipe(
        text=messages,
        images=[image],
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        return_full_text=False,
    )
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        generated = generated[-1].get("content", str(generated[-1]))
    return str(generated)


def score_artwork(image_path, rubric, max_tokens, temperature, top_p, use_local_model):
    if not image_path:
        raise gr.Error("Please upload an image before requesting a critique.")

    prompt = build_prompt(rubric)
    model_name = LOCAL_MODEL if use_local_model else REMOTE_MODEL
    mode = "Local" if use_local_model else "Hosted"

    try:
        if use_local_model:
            critique = local_score(image_path, prompt, max_tokens, temperature, top_p)
        else:
            critique = remote_score(image_path, prompt, max_tokens, temperature, top_p)
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"The {mode.lower()} model could not score this image: {error}") from error

    return critique, f"Critique complete · {mode} model · `{model_name}`"


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


with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Canvas Critic") as demo:
    image_path_state = gr.State()

    gr.Markdown("# 🎨 Canvas Critic", elem_id="app-title")
    gr.Markdown(
        "Upload an artwork and receive a focused, AI-assisted score and critique.",
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
                rubric = gr.Textbox(
                    value=DEFAULT_RUBRIC,
                    label="Critic instructions",
                    lines=8,
                )
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=512,
                    step=32,
                    label="Maximum response tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.4,
                    step=0.1,
                    label="Temperature",
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
                    info="Runs a smaller vision model in this Space instead of the hosted API.",
                )
                gr.Markdown(
                    "Local mode trades some critique quality for privacy and independence "
                    "from the hosted inference API.",
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
            rubric,
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
    demo.launch()
