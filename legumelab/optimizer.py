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
from scipy.optimize import linprog

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
) -> Blend | None:
    """Cheapest blend meeting the protein and amino-acid-score targets.

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

    bounds = []
    for name in names:
        lo = (min_share or {}).get(name, 0.0)
        hi = (max_share or {}).get(name, 1.0)
        if excluded and name in excluded:
            lo, hi = 0.0, 0.0
        bounds.append((lo, hi))

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
    """Up to k different blends to test in the kitchen.

    After each solution, the main ingredient of that blend is capped lower so
    the next solution is genuinely different, not a near-copy.
    """
    blends: list[Blend] = []
    max_share = dict(kwargs.pop("max_share", None) or {})
    for _ in range(k):
        b = cheapest_blend(ing, ref, min_protein, min_score, max_share=max_share, **kwargs)
        if b is None:
            break
        blends.append(b)
        top = max(b.fractions, key=b.fractions.get)
        max_share[top] = max(0.0, b.fractions[top] - 0.15)
    return blends


if __name__ == "__main__":
    ing, ref = load_data()
    for b in top_distinct_blends(ing, ref, min_protein=15, min_score=0.9):
        print(b.as_row())
