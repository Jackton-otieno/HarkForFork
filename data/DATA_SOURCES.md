# Data sources and provenance

**Every row in `ingredients.csv` is currently `PLACEHOLDER`.** The numbers are rough
illustrative values so the app runs on day one. They are NOT to be shown to the jury.
The app displays a red warning while any placeholder row remains.

Replace them on Thursday night / Friday morning, then change the `source` column to
say exactly where each value came from (e.g. `KFCT2018 code 01023`, `KAMIS Nairobi 2026-11-20`).

| Column | Where to get it | Label in pitch |
|---|---|---|
| protein_g_per_100g | Kenya Food Composition Tables 2018 (FAO + Govt of Kenya) | Sourced |
| lysine / saa / threonine / tryptophan (mg per g protein) | FAO amino-acid content of foods tables, or USDA FoodData Central (convert mg/100g to mg/g protein) | Sourced |
| digestibility | Literature values for true protein digestibility (cite paper per ingredient) | Sourced or Assumed |
| price_kes_per_kg | KAMIS (kamis.kilimo.go.ke) retail prices, or prices you collect at a market | Sourced / Measured |

`saa` = sulphur amino acids (methionine + cysteine).

Reference pattern (`reference_pattern.csv`): FAO/WHO/UNU 2007 adult pattern.
If your target user is a young-child porridge, switch to the child pattern and say so.

Anything produced by the language model (processing steps, explanations) is labelled
"model-generated" in the app.
