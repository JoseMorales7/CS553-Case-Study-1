"""Local critique with ArtiMuse (InternVL3-8B fine-tuned for image aesthetics).

ArtiMuse's real `.score()` lives only in the GitHub repo, not in the Hugging
Face one -- `grep -c "def score"` on the Hub's modeling_internvl_chat.py returns
0. Rather than vendor src/artimuse/ (which drags in peft and transformers-4.x
era internlm2/phi3 files), we use the Hub's `.chat()` with ArtiMuse's own
scoring question and decode the two-letter answer ourselves.

The upstream method takes a softmax-weighted expectation over all 101 score
tokens (e.g. 63.47); decoding the generated token gives the argmax (63). Same
number for our purposes, a fraction of the integration work.
"""

import logging
import os

import spaces
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from config import (
    ARTIMUSE_ASPECT_QUESTION,
    ARTIMUSE_SCORE_QUESTION,
    LOCAL_MODEL,
)
from critique import Critique

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448

# First letter selects the block of 25, second letter is the offset within it.
_TOKEN_BASE = {"a": 0, "c": 26, "d": 51, "e": 76}

_model = None
_tokenizer = None


def aestoken2score(token: str) -> int | None:
    """Decode ArtiMuse's two-letter score code, e.g. 'dc' -> 53.

    Returns None if the model answered with something unexpected, so the caller
    can fail over rather than surface a wrong number.
    """
    token = token.strip().lower()
    if len(token) < 2:
        return None
    first, second = token[0], token[1]
    if first not in _TOKEN_BASE or not ("a" <= second <= "z"):
        return None
    score = _TOKEN_BASE[first] + (ord(second) - ord("a"))
    return score if 0 <= score <= 100 else None


def load_image(image_path: str) -> torch.Tensor:
    """ArtiMuse evaluates a single 448px tile -- no dynamic tiling."""
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(image_path) as image:
        return transform(image).unsqueeze(0)


def _device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        # The checkpoint is bfloat16. float16 has a much smaller exponent range
        # and overflows to inf inside attention, which makes the model emit a
        # garbage token instead of an aesthetic code.
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _artimuse_class():
    """Fetch ArtiMuse's bundled model class, patched for transformers 5.x.

    The repo's modeling code was written against transformers 4.x and never
    calls post_init(), which is where 5.x creates `all_tied_weights_keys`.
    Loading therefore dies in _move_missing_keys_from_meta_to_device. Both the
    top-level and llm configs set tie_word_embeddings=False, so the mapping is
    genuinely empty and this patch is faithful, not a workaround for a real tie.
    """
    cls = get_class_from_dynamic_module(
        "modeling_internvl_chat.InternVLChatModel", LOCAL_MODEL
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    return cls


def _ensure_loaded():
    """Load the 15.9 GB checkpoint, once.

    Inside a Space this is called at import (see the bottom of this module), not
    on first request. Space disk is ephemeral, so the first call after a restart
    would otherwise download 15.9 GB *inside* the @spaces.GPU window and blow the
    duration limit. ZeroGPU also documents that models should be placed on cuda
    at root module level. Locally we stay lazy so importing this module for a
    test does not pull 16 GB into memory.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    device, dtype = _device_and_dtype()
    logger.info("Loading %s on %s (%s)", LOCAL_MODEL, device, dtype)
    _model = (
        _artimuse_class()
        .from_pretrained(
            LOCAL_MODEL,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,  # falls back to eager; no flash-attn build needed
        )
        .eval()
        .to(device)
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL, trust_remote_code=True, use_fast=False
    )
    return _model, _tokenizer


@spaces.GPU(duration=120)
def local_critique(image_path, max_tokens, temperature, top_p) -> Critique:
    model, tokenizer = _ensure_loaded()
    device, dtype = _device_and_dtype()
    pixel_values = load_image(image_path).to(dtype).to(device)

    # Scoring is a single forward pass: we only need the first generated token.
    score_config = dict(max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw_score = model.chat(tokenizer, pixel_values, ARTIMUSE_SCORE_QUESTION, score_config)
    score = aestoken2score(str(raw_score))
    if score is None:
        raise RuntimeError(f"ArtiMuse returned an unparseable score token: {raw_score!r}")

    # The written assessment is the only generative call, so cap it tightly --
    # upstream defaults to max_new_tokens=8192, which would blow the GPU budget.
    text_config = dict(
        max_new_tokens=min(int(max_tokens), 512),
        do_sample=temperature > 0,
        temperature=max(float(temperature), 0.01),
        top_p=float(top_p),
        pad_token_id=tokenizer.eos_token_id,
    )
    evaluation = model.chat(tokenizer, pixel_values, ARTIMUSE_ASPECT_QUESTION, text_config)

    return Critique(
        score=score,
        evaluation=str(evaluation),
        model_name=LOCAL_MODEL,
        route="Local",
    )


# On a Space, pay the download/load cost at boot rather than inside the first
# @spaces.GPU call, which has a hard duration limit. SPACE_ID is set by the
# Spaces runtime and is absent locally.
if os.environ.get("SPACE_ID"):
    _ensure_loaded()
