"""Hugging Face token resolution.

Visitors sign in and are billed on their own token. The only fallback is for
local development: outside a Space, Gradio mocks OAuth with a fixed fake token,
so we substitute the machine's own login. Gradio never mocks inside a Space, so
this fallback cannot leak onto the deployed app.
See gradio/oauth.py::_get_mocked_oauth_info.
"""

from huggingface_hub import get_token

MOCK_OAUTH_TOKEN = "mock-oauth-token-for-local-dev"


def resolve_token(hf_token) -> str | None:
    """Return the visitor's HF token, or None if they are not signed in."""
    token = getattr(hf_token, "token", None)
    if token == MOCK_OAUTH_TOKEN:
        return get_token()
    return token
