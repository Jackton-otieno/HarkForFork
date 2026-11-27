"""Blend optimiser for LegumeLab.

Finds the cheapest blend of ingredients that meets a protein target and a
protein-quality (amino-acid score) target, using linear programming.

Why it is linear: the amino-acid score of a blend is
    min over amino acids a of  [ sum_i x_i * p_i * aa_i,a ] / [ ref_a * sum_i x_i * p_i ]
Requiring score >= s is the same as, for every amino acid a:
    sum_i x_i * p_i * (aa_i,a - s * ref_a) >= 0
which is linear in the blend fractions x_i for a fixed s.
Sweeping s and the protein target gives the cost / quality trade-off curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AMINO_ACIDS = ["lysine", "saa", "threonine", "tryptophan"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    ing = pd.read_csv(DATA_DIR / "ingredients.csv")
    ref = pd.read_csv(DATA_DIR / "reference_pattern.csv").set_index("amino_acid")["mg_per_g_protein"]
    return ing, ref


@dataclass
class Blend:
    fractions: dict[str, float]      # ingredient -> share of the blend (0..1)
    protein_g_per_100g: float
    aa_score: float                  # amino-acid score, capped at 1.0
    pdcaas: float                    # digestibility-corrected score, capped at 1.0
    limiting_amino_acid: str
    cost_kes_per_kg: float

    def as_row(self) -> dict:
        row = {f"{k} %": round(float(v) * 100, 1) for k, v in self.fractions.items() if v > 0.001}
        row.update({
            "protein g/100g": round(float(self.protein_g_per_100g), 1),
            "AA score": round(float(self.aa_score), 2),
            "PDCAAS": round(float(self.pdcaas), 2),
            "limiting AA": self.limiting_amino_acid,
            "cost KES/kg": round(self.cost_kes_per_kg, 0),
        })
        return row


def evaluate(fractions: np.ndarray, ing: pd.DataFrame, ref: pd.Series) -> Blend:
    """Compute the nutrition and cost of any blend (fractions sum to 1)."""
    x = np.asarray(fractions, dtype=float)
    p = ing["protein_g_per_100g"].to_numpy()
    protein_mass = x * p
    total_protein = protein_mass.sum()
    ratios = {}
    for a in AMINO_ACIDS:
        aa = ing[f"{a}_mg_per_g_protein"].to_numpy()
        ratios[a] = (protein_mass * aa).sum() / (total_protein * ref[a])
    limiting = min(ratios, key=ratios.get)
    score = ratios[limiting]
    digest = (protein_mass * ing["digestibility"].to_numpy()).sum() / total_protein
    return Blend(
        fractions=dict(zip(ing["ingredient"], x)),
        protein_g_per_100g=total_protein,
        aa_score=min(score, 1.0),
        pdcaas=min(score * digest, 1.0),
        limiting_amino_acid=limiting,
        cost_kes_per_kg=float((x * ing["price_kes_per_kg"].to_numpy()).sum()),
    )


def cheapest_blend(
    ing: pd.DataFrame,
    ref: pd.Series,
    min_protein: float,
    min_score: float,
    max_share: dict[str, float] | None = None,
    min_share: dict[str, float] | None = None,
    excluded: list[str] | None = None,
    max_legume_share: float | None = None,
    min_used_share: float = 0.0,
) -> Blend | None:
    """Cheapest blend meeting the protein and amino-acid-score targets.

    min_used_share: if > 0, every ingredient is either left out or makes up at least this
    share of the blend (no impractical 1% additions). This needs a yes/no choice per
    ingredient, so it is solved as a mixed-integer program (scipy milp, HiGHS).

    Returns None if no blend can meet the targets.
    """
    n = len(ing)
    names = list(ing["ingredient"])
    p = ing["protein_g_per_100g"].to_numpy()
    cost = ing["price_kes_per_kg"].to_numpy()

    A_ub, b_ub = [], []
    # protein: sum x_i p_i >= min_protein   ->   -sum x_i p_i <= -min_protein
    A_ub.append(-p)
    b_ub.append(-min_protein)
    # amino-acid score >= min_score for each amino acid
    for a in AMINO_ACIDS:
        aa = ing[f"{a}_mg_per_g_protein"].to_numpy()
        A_ub.append(-(p * (aa - min_score * ref[a])))
        b_ub.append(0.0)

    # product-design limit: legumes at most this share of the blend (taste, texture, cooking)
    if max_legume_share is not None:
        A_ub.append((ing["group"] == "legume").to_numpy(dtype=float))
        b_ub.append(max_legume_share)

    bounds = []
    for name in names:
        lo = (min_share or {}).get(name, 0.0)
        hi = (max_share or {}).get(name, 1.0)
        if excluded and name in excluded:
            lo, hi = 0.0, 0.0
        bounds.append((lo, hi))

    if min_used_share <= 0:
        res = linprog(
            c=cost,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=np.ones((1, n)),
            b_eq=[1.0],
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return None
        x = np.clip(res.x, 0, None)
        return evaluate(x / x.sum(), ing, ref)

    # Mixed-integer version: variables [x_1..x_n, y_1..y_n], y_i = 1 if ingredient i is used.
    #   x_i - y_i <= 0                  (unused ingredient -> share 0)
    #   min_used_share * y_i - x_i <= 0 (used ingredient -> share >= min_used_share)
    eye = np.eye(n)
    A = np.vstack([
        np.hstack([np.array(A_ub), np.zeros((len(A_ub), n))]),   # protein, amino acids, legume cap
        np.hstack([eye, -eye]),
        np.hstack([-eye, min_used_share * eye]),
        np.hstack([np.ones((1, n)), np.zeros((1, n))]),          # shares sum to 1
    ])
    lower = np.concatenate([np.full(len(A_ub) + 2 * n, -np.inf), [1.0]])
    upper = np.concatenate([b_ub, np.zeros(2 * n), [1.0]])
    lo_x = [b[0] for b in bounds]
    hi_x = [b[1] for b in bounds]
    res = milp(
        c=np.concatenate([cost, np.zeros(n)]),
        constraints=LinearConstraint(A, lower, upper),
        integrality=np.concatenate([np.zeros(n), np.ones(n)]),
        bounds=Bounds(np.concatenate([lo_x, np.zeros(n)]), np.concatenate([hi_x, np.ones(n)])),
    )
    if not res.success:
        return None
    x = np.clip(res.x[:n], 0, None)
    x[x < 1e-9] = 0.0
    return evaluate(x / x.sum(), ing, ref)


def tradeoff_curve(
    ing: pd.DataFrame,
    ref: pd.Series,
    min_score: float,
    protein_targets: list[float],
    **kwargs,
) -> pd.DataFrame:
    """Cheapest blend for each protein target: the cost vs protein trade-off."""
    rows = []
    for target in protein_targets:
        b = cheapest_blend(ing, ref, min_protein=target, min_score=min_score, **kwargs)
        if b is not None:
            rows.append({"protein target": target, **b.as_row()})
    return pd.DataFrame(rows)


def top_distinct_blends(
    ing: pd.DataFrame,
    ref: pd.Series,
    min_protein: float,
    min_score: float,
    k: int = 3,
    **kwargs,
) -> list[Blend]:
    """Up to k genuinely different blends to test in the kitchen.

    After each solution, its main legume (the protein source) is excluded, so
    the next blend is built on a different legume instead of being a near-copy.
    """
    blends: list[Blend] = []
    excluded = list(kwargs.pop("excluded", None) or [])
    legumes = set(ing.loc[ing["group"] == "legume", "ingredient"])
    for _ in range(k):
        b = cheapest_blend(ing, ref, min_protein, min_score, excluded=excluded, **kwargs)
        if b is None:
            break
        blends.append(b)
        used = {n: v for n, v in b.fractions.items() if n in legumes and v > 0.001}
        if not used:
            break
        excluded.append(max(used, key=used.get))
    return blends


def ingredient_roles(blend: Blend, ing: pd.DataFrame) -> list[dict]:
    """What each ingredient contributes to a blend, so an explanation can be grounded in facts.

    For each ingredient used: its share of the blend, its own protein and price, and the share of
    the blend's protein, lysine and sulphur amino acids (SAA) that it supplies.
    """
    rows = ing.set_index("ingredient")
    used = {n: x for n, x in blend.fractions.items() if x > 0.001}
    prot = {n: x * rows.at[n, "protein_g_per_100g"] for n, x in used.items()}
    lys = {n: prot[n] * rows.at[n, "lysine_mg_per_g_protein"] for n in used}
    saa = {n: prot[n] * rows.at[n, "saa_mg_per_g_protein"] for n in used}
    out = []
    for n, x in used.items():
        out.append({
            "ingredient": n,
            "share_of_blend_pct": round(float(x) * 100, 1),
            "own_protein_g_per_100g": float(rows.at[n, "protein_g_per_100g"]),
            "own_price_kes_per_kg": round(float(rows.at[n, "price_kes_per_kg"]), 1),
            "supplies_pct_of_blend_protein": round(float(100 * prot[n] / sum(prot.values())), 1),
            "supplies_pct_of_blend_lysine": round(float(100 * lys[n] / sum(lys.values())), 1),
            "supplies_pct_of_blend_saa": round(float(100 * saa[n] / sum(saa.values())), 1),
        })
    return out


if __name__ == "__main__":
    ing, ref = load_data()
    for b in top_distinct_blends(ing, ref, min_protein=15, min_score=0.9, max_legume_share=0.4,
                                  min_used_share=0.1):
        print(b.as_row())
