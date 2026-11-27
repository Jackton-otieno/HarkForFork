"""Build data/ingredients.csv from the raw Kenya Food Composition Tables 2018.

Run from the project folder:
    python data/build_ingredients.py

Inputs (all in this repository):
    data/raw/Kenya_Food_Composition_Tables_2018.xlsx   raw file, downloaded from FAO
    data/inputs/ingredient_map.csv                     which KFCT item code = which ingredient
    data/inputs/reference_amino_acids.csv              USDA / Feedipedia values for gaps and checks
    data/raw/FAOSTAT_producer_prices_Kenya_2020-2024.csv  raw FAOSTAT producer prices
    data/inputs/faostat_price_map.csv                  which FAOSTAT item = which ingredient
    data/inputs/retail_prices_reach.csv                retail prices, used only as a cross-check

Outputs:
    data/ingredients.csv        the cleaned table the optimiser reads
    data/cleaning_report.md     every step, every value, every decision

Cleaning steps (numbered the same way in the report):
    1. Read the KFCT data sheet and keep only food rows (rows with a numeric item code).
    2. Pick the KFCT item for each ingredient (ingredient_map.csv).
    3. Clean cell values: "tr" (trace) and blanks -> missing; text numbers -> numbers.
    4. Convert amino acids from mg per 100 g food to mg per g protein.
    5. Where KFCT has no complete amino-acid data, use the reference source instead.
    6. Cross-check KFCT amino acids against the reference source where both exist.
    7. Prices: FAOSTAT producer prices, mean of the last 3 years that all crops share.
    8. Add digestibility, then write ingredients.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "Kenya_Food_Composition_Tables_2018.xlsx"
INPUTS = HERE / "inputs"
OUT_CSV = HERE / "ingredients.csv"
OUT_REPORT = HERE / "cleaning_report.md"
FAOSTAT = HERE / "raw" / "FAOSTAT_producer_prices_Kenya_2020-2024.csv"
N_PRICE_YEARS = 3   # average this many recent years to smooth year-to-year swings

SHEET = "04 USERDATABASE-KFCT2018"
# Column positions in the KFCT data sheet (0-based), checked against its header rows:
COL = {"code": 0, "name": 1, "water": 5, "protein": 6,
       "lysine": 39, "methionine": 40, "cystine": 41, "threonine": 44, "tryptophan": 45}
AMINO = ["lysine", "methionine", "cystine", "threonine", "tryptophan"]

report: list[str] = []


def log(line: str = "") -> None:
    print(line)
    report.append(line)


def clean_number(value):
    """Step 3: KFCT cells can hold numbers, text numbers, 'tr' (trace) or blanks."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("", "tr", "trace", "-", "n/a", "na"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def per_g_protein(values: dict, protein: float) -> dict:
    """Step 4: mg per 100 g food -> mg per g protein, and SAA = Met + Cys."""
    out = {aa: round(values[aa] / protein, 1) for aa in ("lysine", "threonine", "tryptophan")}
    out["saa"] = round((values["methionine"] + values["cystine"]) / protein, 1)
    return out


def load_reference() -> dict:
    refs = {}
    with open(INPUTS / "reference_amino_acids.csv", newline="") as f:
        for r in csv.DictReader(f):
            vals = {aa: float(r[aa]) for aa in AMINO}
            if r["unit"] == "g_per_100g_food":
                p = float(r["protein_g_per_100g"])
                mg = {aa: v * 1000 for aa, v in vals.items()}          # g -> mg per 100 g food
                refs[r["id"]] = (per_g_protein(mg, p), r["source"])
            elif r["unit"] == "g_per_100g_protein":
                mg = {aa: v * 10 for aa, v in vals.items()}            # g/100 g protein -> mg/g protein
                refs[r["id"]] = ({"lysine": round(mg["lysine"], 1),
                                  "saa": round(mg["methionine"] + mg["cystine"], 1),
                                  "threonine": round(mg["threonine"], 1),
                                  "tryptophan": round(mg["tryptophan"], 1)}, r["source"])
            else:
                raise ValueError(f"Unknown unit {r['unit']} for {r['id']}")
    return refs


def build_prices(log) -> dict:
    """Step 7: FAOSTAT producer prices (KES per tonne) -> KES per kg for each ingredient."""
    with open(FAOSTAT, newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.DictReader(f)
               if r["Element"].startswith("Producer Price (LCU/tonne)") and r["Months"] == "Annual value"]
    with open(INPUTS / "faostat_price_map.csv", newline="") as f:
        pmap = list(csv.DictReader(f))
    with open(INPUTS / "retail_prices_reach.csv", newline="") as f:
        retail = {r["ingredient"]: float(r["retail_kes_per_kg"]) for r in csv.DictReader(f)}
    series = {}
    for r in raw:
        series.setdefault(r["Item"], {})[int(r["Year"])] = (float(r["Value"]), r["Flag Description"])
    items = [m["faostat_item"] for m in pmap]
    missing = [i for i in items if i not in series]
    assert not missing, f"FAOSTAT file has no rows for {missing}"
    common = sorted(set.intersection(*(set(series[i]) for i in items)))
    years = common[-N_PRICE_YEARS:]
    all_years = sorted({y for i in items for y in series[i]})

    log("## Step 7 - Prices from FAOSTAT producer prices")
    log("")
    log(f"Source: FAOSTAT, Producer Prices domain, Kenya, element Producer Price (LCU/tonne), annual "
        f"values, file `{FAOSTAT.relative_to(HERE.parent)}`. LCU for Kenya = KES. Years where all "
        f"{len(items)} crops have a price: {common}. Price used = mean of {years}, divided by 1000 "
        "(tonne -> kg) and by the conversion factor.")
    log("")
    head = " | ".join(str(y) for y in all_years)
    log(f"| Ingredient | FAOSTAT item | {head} (KES/tonne) | Mean used | Factor | **KES/kg** | Retail cross-check (REACH) |")
    log("|---|---|" + "---|" * len(all_years) + "---|---|---|---|")
    out = {}
    for m in pmap:
        ser = series[m["faostat_item"]]
        flags = {ser[y][1] for y in years}
        mean_t = sum(ser[y][0] for y in years) / len(years)
        factor = float(m["factor"])
        kes_kg = round(mean_t / 1000 / factor, 1)
        cells = " | ".join(f"{ser[y][0]:,.0f}" if y in ser else "-" for y in all_years)
        rt = retail.get(m["ingredient"])
        rt_txt = f"{rt:.0f} (x{rt / kes_kg:.1f})" if rt else "-"
        log(f"| {m['ingredient']} | {m['faostat_item']} | {cells} | {mean_t:,.0f} | {factor} | **{kes_kg}** | {rt_txt} |")
        note = f"; {m['factor_note']}" if m["factor_note"] else ""
        out[m["ingredient"]] = (kes_kg, f"FAOSTAT producer price, Kenya, {m['faostat_item']}, mean "
                                        f"{years[0]}-{years[-1]} ({', '.join(sorted(flags))}){note}")
    log("")
    log("Producer (farm-gate) prices are lower than shop prices: the REACH column shows retail as a "
        "multiple of the FAOSTAT price. All 8 ingredients use the same basis, so the comparison between "
        "them is fair. Our user is a miller who buys grain, so farm-gate prices are the starting point "
        "for raw-material cost; the app lets the user type in their own prices.")
    log("")
    return out


def main() -> None:
    log("# Data cleaning report")
    log("")
    log(f"Generated by `data/build_ingredients.py` from `{RAW.relative_to(HERE.parent)}`.")
    log("")

    # ---- Step 1
    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    ws = wb[SHEET]
    all_rows = list(ws.iter_rows(values_only=True))
    header = all_rows[2]
    for key, idx in COL.items():  # make sure the columns are where we think they are
        assert key[:3].lower() in str(header[idx]).lower() or key in ("code", "name"), (key, header[idx])
    foods = {r[COL["code"]]: r for r in all_rows if isinstance(r[COL["code"]], int) and r[COL["code"]] > 100}
    log("## Step 1 - Read the KFCT data sheet")
    log(f"Sheet `{SHEET}`: {len(all_rows)} rows in total, {len(foods)} food rows kept "
        "(rows with a numeric item code; group headings and notes dropped). "
        "Values are per 100 g edible portion.")
    log("")

    refs = load_reference()
    with open(INPUTS / "ingredient_map.csv", newline="") as f:
        mapping = list(csv.DictReader(f))
    price_lines: list[str] = []
    prices = build_prices(price_lines.append)

    # ---- Steps 2-5
    log("## Steps 2-5 - Pick items, clean values, convert, fill gaps")
    log("")
    log("| Ingredient | KFCT protein item | Protein g/100 g | Raw KFCT amino acids (mg/100 g: Lys, Met, Cys, Thr, Trp) | Amino-acid source used |")
    log("|---|---|---|---|---|")
    out_rows, checks = [], []
    for m in mapping:
        prow = foods[int(m["kfct_protein_code"])]
        protein = clean_number(prow[COL["protein"]])
        aa_used, aa_source, raw_txt = None, None, "no amino-acid item mapped"
        if m["kfct_aa_code"]:
            arow = foods[int(m["kfct_aa_code"])]
            raw = {aa: clean_number(arow[COL[aa]]) for aa in AMINO}
            raw_txt = ", ".join("missing" if raw[a] is None else f"{raw[a]:.0f}" for a in AMINO)
            if all(v is not None for v in raw.values()):
                a_protein = clean_number(arow[COL["protein"]])
                aa_used = per_g_protein(raw, a_protein)
                aa_source = f"KFCT 2018 code {m['kfct_aa_code']} {str(arow[COL['name']]).strip()}"
        else:
            raw = {aa: clean_number(prow[COL[aa]]) for aa in AMINO}
            raw_txt = ", ".join("missing" if raw[a] is None else f"{raw[a]:.0f}" for a in AMINO)
        ref_vals, ref_src = refs[m["aa_fallback_id"]]
        if aa_used is None:
            aa_used = ref_vals
            aa_source = f"{ref_src} (KFCT has no complete amino-acid data for this food)"
        else:
            checks.append((m["ingredient"], aa_used, ref_vals, ref_src))
        log(f"| {m['ingredient']} | {m['kfct_protein_code']} {str(prow[COL['name']]).strip()} | {protein} "
            f"| {raw_txt} | {aa_source} |")
        price_kes, price_src = prices[m["ingredient"]]
        out_rows.append({
            "ingredient": m["ingredient"], "group": m["group"],
            "protein_g_per_100g": protein,
            "lysine_mg_per_g_protein": aa_used["lysine"], "saa_mg_per_g_protein": aa_used["saa"],
            "threonine_mg_per_g_protein": aa_used["threonine"], "tryptophan_mg_per_g_protein": aa_used["tryptophan"],
            "digestibility": float(m["digestibility"]),
            "price_kes_per_kg": price_kes,
            "source": f"KFCT 2018 code {m['kfct_protein_code']} {str(prow[COL['name']]).strip()}",
            "aa_source": aa_source,
            "price_source": price_src,
        })
    log("")
    log("Step 3 rule: 'tr' (trace) and empty cells are treated as missing. If any of the five amino "
        "acids is missing, the whole KFCT amino-acid profile for that food is replaced by the reference "
        "source (step 5), so profiles are never mixed within one food.")
    log("")
    log("Step 4 formula: mg per g protein = mg amino acid per 100 g food / g protein per 100 g food. "
        "SAA (sulphur amino acids) = methionine + cystine.")
    log("")

    # ---- Step 6
    log("## Step 6 - Cross-check KFCT against the reference source (mg per g protein)")
    log("")
    log("| Ingredient | Amino acid | KFCT | Reference | Difference |")
    log("|---|---|---|---|---|")
    for ing, k, r, src in checks:
        for aa in ("lysine", "saa", "threonine", "tryptophan"):
            diff = (k[aa] - r[aa]) / r[aa] * 100
            flag = " **check**" if abs(diff) > 30 else ""
            log(f"| {ing} | {aa} | {k[aa]} | {r[aa]} | {diff:+.0f}%{flag} |")
    log("")
    log("Differences above 30% are marked **check**. KFCT values are kept (hackathon-recommended, "
        "partly lab-measured for KFCT 2018); see the sensitivity test in DATA_SOURCES.md.")
    log("")

    # ---- Step 7 (computed earlier, reported here in order)
    for line in price_lines:
        log(line)

    # ---- Step 8
    fields = list(out_rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    log("## Step 8 - Output")
    log(f"Wrote `{OUT_CSV.relative_to(HERE.parent)}` with {len(out_rows)} ingredients. "
        "Digestibility values are ASSUMED (see DATA_SOURCES.md).")
    OUT_REPORT.write_text("\n".join(report) + "\n")
    print(f"\nReport written to {OUT_REPORT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
