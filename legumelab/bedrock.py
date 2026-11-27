"""Thin wrapper around Amazon Bedrock for LegumeLab.

- explain_blend(): plain-language explanation of a blend (Converse API)
- ask_knowledge_base(): grounded answer with citations from a Bedrock Knowledge Base

Credentials come from the environment (paste the temporary AWS CLI credentials
from the Workshop Studio event page into your terminal). Never hard-code keys.
"""
from __future__ import annotations

import json
import re
import os
import time
from functools import lru_cache

import boto3

REGION = "us-east-1"
# Copy the exact model / inference-profile ID from the Bedrock console
# (Model catalog -> model -> "Model ID" or Cross-region inference -> "Inference profile ID").
MODEL_ID = os.environ.get("LEGUMELAB_MODEL_ID", "PASTE_MODEL_ID_HERE")
KB_ID = os.environ.get("LEGUMELAB_KB_ID", "")  # set after you create the Knowledge Base

_last_call = 0.0


def _rate_limit(min_gap: float = 1.0) -> None:
    """Stay under ~1 request/second, as the AWS participant guide asks."""
    global _last_call
    wait = min_gap - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


@lru_cache(maxsize=1)
def _runtime():
    return boto3.client("bedrock-runtime", region_name=REGION)


@lru_cache(maxsize=1)
def _agent_runtime():
    return boto3.client("bedrock-agent-runtime", region_name=REGION)


SYSTEM_PROMPT = (
    "You are a food-formulation assistant for a small Kenyan flour/porridge company. "
    "You explain blends computed by an optimiser. Use ONLY the numbers given to you; "
    "never invent nutrition values. Be concise and practical. "
    "Explain: why these ingredients complement each other (amino acids), the trade-off "
    "versus the other blends, and 3-5 processing steps suited to a small Kenyan mill "
    "(cleaning, roasting/soaking, milling, sieving, packing). "
    "Clearly mark processing advice as general guidance to be validated in a kitchen trial. "
    "Format: no headings larger than ###, at most 250 words. "
    "Never claim effects (e.g. of processing on digestibility) that are not in the numbers you were given. "
    "Compare blends ONLY on the fields provided (shares, protein, AA score, PDCAAS, limiting AA, cost). "
    "Do not mention shelf life, taste, texture, carbohydrates or any property not in those fields. "
    "Before stating that one blend costs more or less than another, check the cost numbers. "
    "To explain why an ingredient is in the blend, use ONLY ingredient_roles: its share of the "
    "blend's protein, lysine and SAA, its own protein and its price. Do not give other reasons."
)

# Topics we have no data on; if the model mentions one, we ask it once to rewrite.
UNSUPPORTED = ["mineral", "vitamin", "iron", "zinc", "fibre", "fiber", "carbohydrate", "energy",
               "calorie", "shelf", "taste", "flavour", "flavor", "texture", "colour", "color",
               "aroma", "allerg", "glycemic", "glycaemic", "antioxidant", "gluten"]


def unsupported_terms(text: str) -> list[str]:
    low = text.lower()
    return [t for t in UNSUPPORTED if re.search(r"\b" + t, low)]


def tidy_headings(text: str) -> str:
    """Streamlit renders # and ## very large; demote them to ####."""
    return re.sub(r"^#{1,3} ", "#### ", text, flags=re.M)


def _converse(messages: list[dict]) -> str:
    _rate_limit()
    resp = _runtime().converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": 700, "temperature": 0.1},
    )
    return resp["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=256)
def _explain_cached(payload: str) -> tuple[str, tuple[str, ...]]:
    """Returns (text, unsupported terms still present after one rewrite)."""
    messages = [{"role": "user", "content": [{"text": payload}]}]
    text = _converse(messages)
    bad = unsupported_terms(text)
    if bad:  # one rewrite, telling the model exactly what to remove
        messages += [
            {"role": "assistant", "content": [{"text": text}]},
            {"role": "user", "content": [{"text": (
                "Rewrite your answer. Remove every statement about: " + ", ".join(bad) +
                ". We have no data on these. Keep everything else, use only the data given.")}]},
        ]
        text = _converse(messages)
        bad = unsupported_terms(text)
    return tidy_headings(text), tuple(bad)


def explain_blend(blend_row: dict, alternatives: list[dict], user_goal: str,
                  roles: list[dict] | None = None) -> tuple[str, list[str]]:
    """Plain-language explanation of a blend.

    Returns (markdown text, list of unsupported topics the model still mentioned).
    """
    payload = json.dumps(
        {"user_goal": user_goal, "chosen_blend": blend_row, "ingredient_roles": roles or [],
         "other_blends": alternatives},
        sort_keys=True,
    )
    text, bad = _explain_cached(payload)  # cached: same input -> no second model call
    return text, list(bad)


def ask_knowledge_base(question: str) -> tuple[str, list[str]]:
    """Grounded answer from your Knowledge Base (processing guides, standards).

    Returns (answer, list of source document URIs).
    """
    if not KB_ID:
        return "Knowledge Base not configured yet (set LEGUMELAB_KB_ID).", []
    _rate_limit()
    model_arn = MODEL_ID if MODEL_ID.startswith("arn:") else (
        f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"
    )
    resp = _agent_runtime().retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {"knowledgeBaseId": KB_ID, "modelArn": model_arn},
        },
    )
    sources = []
    for c in resp.get("citations", []):
        for ref in c.get("retrievedReferences", []):
            uri = ref.get("location", {}).get("s3Location", {}).get("uri")
            if uri and uri not in sources:
                sources.append(uri)
    return resp["output"]["text"], sources
