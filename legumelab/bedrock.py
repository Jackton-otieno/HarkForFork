"""Thin wrapper around Amazon Bedrock for LegumeLab.

- explain_blend(): plain-language explanation of a blend (Converse API)
- ask_knowledge_base(): grounded answer with citations from a Bedrock Knowledge Base

Credentials come from the environment (paste the temporary AWS CLI credentials
from the Workshop Studio event page into your terminal). Never hard-code keys.
"""
from __future__ import annotations

import re
import os
import time
from functools import lru_cache

import boto3

REGION = "us-east-1"
# Copy the exact model / inference-profile ID from the Bedrock console
# (Model catalog -> model -> "Model ID" or Cross-region inference -> "Inference profile ID").
MODEL_ID = os.environ.get("LEGUMELAB_MODEL_ID", "PASTE_MODEL_ID_HERE")
GUARDRAIL_ID = os.environ.get("LEGUMELAB_GUARDRAIL_ID", "")  # legumelab-grounding guardrail
GUARDRAIL_VERSION = os.environ.get("LEGUMELAB_GUARDRAIL_VERSION", "DRAFT")
KB_ID = os.environ.get("LEGUMELAB_KB_ID", "LNZHEFARUI")  # legumelab-processing-kb (us-east-1)

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
    "Explain: why these ingredients complement each other (amino acids) and the trade-off "
    "versus the other blends. Do NOT give processing steps: the app shows processing "
    "guidance separately, from cited documents. "
    "Format: no headings larger than ###, at most 250 words. "
    "Never claim effects (e.g. of processing on digestibility) that are not in the numbers you were given. "
    "Compare blends ONLY on the fields provided (shares, protein, AA score, PDCAAS, limiting AA, cost). "
    "Do not mention shelf life, taste, texture, carbohydrates or any property not in those fields. "
    "Before stating that one blend costs more or less than another, check the cost numbers. "
    "To explain why an ingredient is in the blend, use ONLY its listed share of the blend's "
    "protein, lysine and SAA, its own protein and its price. Do not give other reasons. "
    "You are given a fact sheet. State only facts from it; you may quote cost differences "
    "that appear in it, but do not calculate new percentages or ratios. Every blend listed "
    "meets every constraint; never say a blend breaks a constraint."
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


def _pct(row: dict) -> str:
    return ", ".join(f"{k[:-2].replace('_', ' ')} {v:g}%" for k, v in row.items()
                     if k.endswith(" %") and v)


def fact_sheet(blend_row: dict, alternatives: list[dict], user_goal: str,
               roles: list[dict] | None = None) -> str:
    """The data the model may use, as plain sentences (Guardrails grounds best on prose)."""
    def describe(name: str, r: dict) -> str:
        return (f"{name} is {_pct(r)}. It costs {r['cost KES/kg']:g} KES per kg, has "
                f"{r['protein g/100g']:g} g protein per 100 g, an amino-acid score of "
                f"{r['AA score']:g}, a PDCAAS of {r['PDCAAS']:g}, and its limiting amino acid "
                f"is {r['limiting AA']}.")
    lines = [f"The user's targets are: {user_goal}.",
             "The optimiser found the cheapest blends that meet every target. "
             "Every blend listed below meets every target and constraint.",
             describe("The chosen blend", blend_row)]
    for i, alt in enumerate(alternatives, 1):
        diff = alt["cost KES/kg"] - blend_row["cost KES/kg"]
        lines.append(describe(f"Alternative blend {i}", alt) +
                     f" It costs {abs(diff):g} KES per kg {'more' if diff > 0 else 'less'} "
                     "than the chosen blend." if diff else describe(f"Alternative blend {i}", alt) +
                     " It costs the same as the chosen blend.")
    for r in roles or []:
        n = r["ingredient"].replace("_", " ")
        lines.append(
            f"In the chosen blend, {n} is {r['share_of_blend_pct']:g}% of the blend and supplies "
            f"{r['supplies_pct_of_blend_protein']:g}% of the blend's protein, "
            f"{r['supplies_pct_of_blend_lysine']:g}% of its lysine and "
            f"{r['supplies_pct_of_blend_saa']:g}% of its sulphur amino acids (SAA). "
            f"{n.capitalize()} has {r['own_protein_g_per_100g']:g} g protein per 100 g and costs "
            f"{r['own_price_kes_per_kg']:g} KES per kg.")
    lines.append("SAA means methionine plus cysteine. PDCAAS is protein quality corrected for "
                 "digestibility; 1.0 is the maximum.")
    return "\n".join(lines)


def verified_summary(blend_row: dict, alternatives: list[dict],
                     roles: list[dict] | None = None) -> str:
    """A summary written by code straight from the optimiser's numbers: correct by construction.
    Shown instead of the AI text when that text fails the Guardrails grounding check."""
    lines = [f"**Chosen blend:** {_pct(blend_row)}. **{blend_row['cost KES/kg']:g} KES/kg**, "
             f"{blend_row['protein g/100g']:g} g protein per 100 g, amino-acid score "
             f"{blend_row['AA score']:g}, PDCAAS {blend_row['PDCAAS']:g}, limiting amino acid: "
             f"{blend_row['limiting AA']}.", ""]
    for r in sorted(roles or [], key=lambda r: -r["supplies_pct_of_blend_protein"]):
        lines.append(f"- **{r['ingredient'].replace('_', ' ').capitalize()}** "
                     f"({r['share_of_blend_pct']:g}% of the blend, {r['own_price_kes_per_kg']:g} KES/kg) "
                     f"supplies {r['supplies_pct_of_blend_protein']:g}% of the protein, "
                     f"{r['supplies_pct_of_blend_lysine']:g}% of the lysine and "
                     f"{r['supplies_pct_of_blend_saa']:g}% of the sulphur amino acids.")
    if alternatives:
        lines.append("")
        for i, a in enumerate(alternatives, 1):
            diff = a["cost KES/kg"] - blend_row["cost KES/kg"]
            cmp = f"{abs(diff):g} KES/kg {'more' if diff > 0 else 'less'}" if diff else "the same cost"
            lines.append(f"- **Alternative {i}** ({_pct(a)}): {cmp}; {a['protein g/100g']:g} g protein, "
                         f"PDCAAS {a['PDCAAS']:g}.")
    return "\n".join(lines)


def _converse(messages: list[dict]) -> str:
    """One Converse call. Some newer models (e.g. Claude Sonnet 5) reject `temperature`;
    for those we retry once without it."""
    _rate_limit()
    args = dict(modelId=MODEL_ID, system=[{"text": SYSTEM_PROMPT}], messages=messages)
    try:
        resp = _runtime().converse(**args, inferenceConfig={"maxTokens": 700, "temperature": 0.1})
    except Exception as e:
        if "temperature" not in str(e):
            raise
        _rate_limit()
        resp = _runtime().converse(**args, inferenceConfig={"maxTokens": 700})
    return resp["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=256)
def _explain_cached(facts: str, query: str) -> tuple[str, tuple[str, ...], tuple]:
    """Returns (text, unsupported terms left, Guardrails grounding results).

    At most one rewrite: if the answer mentions topics we have no data on, or Guardrails
    scores it below the grounding threshold, Claude is asked once to fix it.
    """
    messages = [{"role": "user", "content": [{"text": "Fact sheet:\n" + facts}]}]
    text = _converse(messages)
    bad = unsupported_terms(text)
    grounding = _safe_grounding(facts, query, text)
    failed = [g for g in grounding if not g[3] and not g[0].startswith("error")]
    if bad or failed:
        problems = []
        if bad:
            problems.append("remove every statement about " + ", ".join(bad) + " (no data on these)")
        if failed:
            problems.append("some statements are not supported by the fact sheet; keep only "
                            "statements that repeat facts from it, and do not calculate new numbers")
        messages += [
            {"role": "assistant", "content": [{"text": text}]},
            {"role": "user", "content": [{"text": "Rewrite your answer: " + "; ".join(problems) + "."}]},
        ]
        text = _converse(messages)
        bad = unsupported_terms(text)
        grounding = _safe_grounding(facts, query, text)
    return tidy_headings(text), tuple(bad), grounding


def _safe_grounding(facts: str, query: str, text: str) -> tuple:
    try:
        return grounding_check(facts, query, text)
    except Exception as e:  # the explanation still shows if the guardrail call fails
        return (("error: " + str(e)[:120], 0.0, 0.0, True),)


@lru_cache(maxsize=128)
def grounding_check(source: str, query: str, answer: str) -> tuple[tuple[str, float, float, bool], ...]:
    """Amazon Bedrock Guardrails contextual grounding check.

    Scores how well `answer` is supported by `source` (grounding) and answers `query` (relevance).
    Returns ((type, score, threshold, passed), ...); empty if no guardrail is configured.
    """
    if not GUARDRAIL_ID:
        return ()
    _rate_limit()
    resp = _runtime().apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="OUTPUT",
        content=[
            {"text": {"text": source, "qualifiers": ["grounding_source"]}},
            {"text": {"text": query, "qualifiers": ["query"]}},
            {"text": {"text": answer, "qualifiers": ["guard_content"]}},
        ],
    )
    out = []
    for a in resp.get("assessments", []):
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            out.append((f["type"].lower(), float(f["score"]), float(f["threshold"]),
                        f.get("action") != "BLOCKED"))
    return tuple(out)


def explain_blend(blend_row: dict, alternatives: list[dict], user_goal: str,
                  roles: list[dict] | None = None) -> tuple[str, list[str], tuple]:
    """Plain-language explanation of a blend.

    Returns (markdown text, unsupported topics the model still mentioned,
             Guardrails grounding results - empty if no guardrail is configured).
    """
    facts = fact_sheet(blend_row, alternatives, user_goal, roles)
    text, bad, grounding = _explain_cached(facts, "Explain why this blend was chosen: " + user_goal)
    return text, list(bad), grounding

@lru_cache(maxsize=1)
def _account_id() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _model_arn() -> str:
    """Knowledge Bases need a full ARN. Inference profiles (us., global.) have their own ARN form."""
    if MODEL_ID.startswith("arn:"):
        return MODEL_ID
    if MODEL_ID.split(".")[0] in ("us", "global", "eu", "apac"):
        return f"arn:aws:bedrock:{REGION}:{_account_id()}:inference-profile/{MODEL_ID}"
    return f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"


@lru_cache(maxsize=64)
def ask_knowledge_base(question: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Grounded answer from the Bedrock Knowledge Base (processing documents in S3).

    Returns (answer, sources) where each source is (file name, short excerpt it was based on).
    Cached: the same question does not call Bedrock twice.
    """
    if not KB_ID:
        return "Knowledge Base not configured (set LEGUMELAB_KB_ID).", ()
    _rate_limit()
    resp = _agent_runtime().retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {"knowledgeBaseId": KB_ID, "modelArn": _model_arn()},
        },
    )
    sources, seen = [], set()
    for c in resp.get("citations", []):
        for ref in c.get("retrievedReferences", []):
            uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
            excerpt = " ".join(ref.get("content", {}).get("text", "").split())[:320]
            key = (uri, excerpt[:80])
            if uri and key not in seen:
                seen.add(key)
                sources.append((uri.rsplit("/", 1)[-1], excerpt))
    return resp["output"]["text"], tuple(sources)


def processing_question(ingredient_names: list[str]) -> str:
    return (
        "A small mill in Kenya makes a porridge flour from " + ", ".join(ingredient_names) + ". "
        "Using only the documents, how should these grains and legumes be prepared before and "
        "during milling (for example cleaning, soaking, dehulling, roasting, germination, milling), "
        "and why does each step matter for protein quality or anti-nutritional factors? "
        "Answer as 4 to 6 short numbered steps, under 200 words. If the documents do not cover "
        "an ingredient, say so."
    )
