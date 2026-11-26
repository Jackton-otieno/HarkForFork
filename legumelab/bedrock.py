"""Thin wrapper around Amazon Bedrock for LegumeLab.

- explain_blend(): plain-language explanation of a blend (Converse API)
- ask_knowledge_base(): grounded answer with citations from a Bedrock Knowledge Base

Credentials come from the environment (paste the temporary AWS CLI credentials
from the Workshop Studio event page into your terminal). Never hard-code keys.
"""
from __future__ import annotations

import json
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
    "Clearly mark processing advice as general guidance to be validated in a kitchen trial."
)


@lru_cache(maxsize=256)
def _explain_cached(payload: str) -> str:
    _rate_limit()
    resp = _runtime().converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": payload}]}],
        inferenceConfig={"maxTokens": 700, "temperature": 0.3},
    )
    return resp["output"]["message"]["content"][0]["text"]


def explain_blend(blend_row: dict, alternatives: list[dict], user_goal: str) -> str:
    payload = json.dumps(
        {"user_goal": user_goal, "chosen_blend": blend_row, "other_blends": alternatives},
        sort_keys=True,
    )
    return _explain_cached(payload)  # cached: same input -> no second model call


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
