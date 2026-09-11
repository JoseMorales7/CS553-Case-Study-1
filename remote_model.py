"""Hosted critique via the Hugging Face Inference Providers API."""

import base64
import mimetypes
import re
from pathlib import Path

from huggingface_hub import InferenceClient

from config import EVALUATION_HEADING, REMOTE_MODEL, REMOTE_PROMPT, REMOTE_PROVIDER
from critique import Critique

SCORE_PATTERN = re.compile(r"##\s*Score:\s*(\d+)\s*/\s*100")
HEADING_PATTERN = re.compile(rf"###\s*{re.escape(EVALUATION_HEADING)}\s*", re.IGNORECASE)


def image_as_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_response(text: str) -> tuple[int | None, str]:
    """Split the model's markdown into (score, evaluation body).

    Qwen3 intermittently emits the whole answer twice (observed in roughly 1 run
    in 4), so anything after a second "## Score" heading is discarded.
    """
    headings = list(SCORE_PATTERN.finditer(text))
    if len(headings) > 1:
        text = text[: headings[1].start()]

    score = int(headings[0].group(1)) if headings else None

    body = HEADING_PATTERN.split(text, maxsplit=1)
    evaluation = body[1].strip() if len(body) > 1 else text.strip()

    # A repeat can also start straight at the advice heading rather than the
    # score, so cut there too.
    repeat = HEADING_PATTERN.search(evaluation)
    if repeat:
        evaluation = evaluation[: repeat.start()].strip()

    return score, evaluation


def remote_critique(image_path, max_tokens, temperature, top_p, hf_token) -> Critique:
    client = InferenceClient(provider=REMOTE_PROVIDER, token=hf_token)
    response = client.chat.completions.create(
        model=REMOTE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_as_data_url(image_path)}},
                    {"type": "text", "text": REMOTE_PROMPT},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    # Qwen3 is a reasoning model: `max_tokens` covers its internal thinking as
    # well as the answer, so a tight budget yields an empty response with no
    # exception. Raise so the caller's failover can catch it.
    if not content:
        raise RuntimeError(
            f"Hosted model returned no visible content (finish_reason="
            f"{choice.finish_reason}); the token budget was likely exhausted by "
            "internal reasoning. Try raising 'Maximum response tokens'."
        )

    score, evaluation = parse_response(content)
    return Critique(score=score, evaluation=evaluation, model_name=REMOTE_MODEL, route="Hosted")
