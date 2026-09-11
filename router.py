"""Chooses which model handles a request, and what happens when one fails."""

import logging

import gradio as gr

from config import LOCAL_MODEL, REMOTE_MODEL
from critique import Critique
from hf_auth import resolve_token
from local_model import local_critique
from remote_model import remote_critique

logger = logging.getLogger(__name__)


def score_artwork(
    image_path,
    max_tokens,
    temperature,
    top_p,
    use_local_model,
    hf_token: gr.OAuthToken | None,
):
    """Return (markdown, status) for the UI.

    The annotation on hf_token MUST stay Optional. With a bare gr.OAuthToken,
    Gradio rejects logged-out visitors before this function runs, and the
    failover below never gets a chance to fire.
    """
    if not image_path:
        raise gr.Error("Please upload an image before requesting a critique.")

    try:
        if use_local_model:
            result = _local_with_fallback(image_path, max_tokens, temperature, top_p, hf_token)
            return result.to_markdown(), result.to_status()

        token = resolve_token(hf_token)
        if not token:
            gr.Warning(
                "No Hugging Face credentials available. "
                "Automatically switching to the local model."
            )
            result = local_critique(image_path, max_tokens, temperature, top_p)
            result.route = "Failover (no credentials)"
            return result.to_markdown(), result.to_status()

        try:
            result = remote_critique(image_path, max_tokens, temperature, top_p, token)
            return result.to_markdown(), result.to_status()
        except Exception:
            logger.warning("Hosted model failed, failing over to local", exc_info=True)
            gr.Warning(
                "The hosted model is unavailable. "
                "Automatically switching to the local model."
            )
            result = local_critique(image_path, max_tokens, temperature, top_p)
            result.route = "Failover (hosted unavailable)"
            return result.to_markdown(), result.to_status()

    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"The model could not score this image: {error}") from error


def _local_with_fallback(image_path, max_tokens, temperature, top_p, hf_token) -> Critique:
    """Local mode, with one escape hatch.

    ArtiMuse runs on ZeroGPU, whose daily quota is small (2 minutes for an
    anonymous visitor). When it runs out, @spaces.GPU raises and the user would
    otherwise just see an error. Delete this fallback if you would rather the
    failure be visible.
    """
    try:
        return local_critique(image_path, max_tokens, temperature, top_p)
    except Exception:
        token = resolve_token(hf_token)
        if not token:
            raise
        logger.warning("Local model failed, falling back to hosted", exc_info=True)
        gr.Warning(
            "The local model is unavailable (GPU quota may be exhausted). "
            "Falling back to the hosted model."
        )
        result = remote_critique(image_path, max_tokens, temperature, top_p, token)
        result.route = "Failover (local unavailable)"
        return result
