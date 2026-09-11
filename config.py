"""Model identifiers, prompts and the shared output contract.

Both the hosted and the local model must produce the same markdown shape so the
UI (and the report's comparison) treats them identically:

    ## Score: N / 100

    ### How to improve
    - <concrete change>
"""

REMOTE_MODEL = "Qwen/Qwen3.8-27B"
LOCAL_MODEL = "Thunderbolt215215/ArtiMuse"

# Pinning a provider is a cost decision, not a performance one. As of this
# writing ovhcloud serves this model at $0/M tokens while deepinfra and novita
# charge ~$0.40/M in and ~$3.00/M out. "auto" may pick either.
REMOTE_PROVIDER = "auto"

# The heading both models' advice is rendered under.
EVALUATION_HEADING = "How to improve"

# ArtiMuse's own scoring question. The model is trained to answer with a
# two-letter code that maps onto 0-100; see local_model.aestoken2score.
ARTIMUSE_SCORE_QUESTION = """
Rate the aesthetics score of the image in 0-100.
In the output format, numbers are replaced by 2 corresponding letters,
and the mapping relationship is:
score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

The answer only outputs 2 corresponding letters.
"""

# ArtiMuse ships eight canned attribute questions of the form "Please evaluate
# the aesthetic quality of this image from the aspect of <X>". Tested on the
# Space, those return evaluative description -- accurate, but it tells the user
# what their image looks like, which they can already see. ArtiMuse is an
# InternVL3-8B chat model, so we are not limited to those questions; asking for
# advice directly gets advice.
ARTIMUSE_ADVICE_QUESTION = (
    "Suggest three specific ways to improve the aesthetic quality of this image. "
    "Consider composition, lighting, colour, and technique. "
    "Answer as exactly three short bullet points, each naming one concrete change "
    "the artist should make. Do not describe what the image shows."
)

# The hosted model has no aesthetic head, so we ask it for the same shape in
# words. Kept deliberately terse: the earlier paragraph-length version described
# the image back to the user, which they can already see.
REMOTE_PROMPT = f"""Rate this image's aesthetic quality.

Reply in exactly this format, once, and nothing else:

## Score: N / 100

### {EVALUATION_HEADING}
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>

Rules:
- N is a whole number from 0 to 100, judging composition, lighting, colour and
  technique.
- Do NOT describe what the image shows. The user can already see it.
- Do NOT justify the score or add any prose outside the three bullets.
- Each bullet must name an action the artist can take.
- Output the format once. Do not repeat it."""
