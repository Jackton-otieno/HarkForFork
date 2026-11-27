# Data sources and provenance

**How the data was cleaned:** run `python data/build_ingredients.py`. It reads the raw
`data/raw/Kenya_Food_Composition_Tables_2018.xlsx`, applies 7 documented steps, writes
`data/ingredients.csv` and a step-by-step `data/cleaning_report.md`. Inputs it uses:
`data/inputs/ingredient_map.csv`, `reference_amino_acids.csv`, `faostat_price_map.csv`,
`retail_prices_reach.csv`, and the raw FAOSTAT file in `data/raw/`.

As of 27 Nov 2026. Each row of `ingredients.csv` names its sources in three columns:
`source` (protein), `aa_source` (amino acids), `price_source` (price).
The app warns while any PLACEHOLDER remains.

## Protein - Sourced (Kenya Food Composition Tables 2018)
All 8 protein values come from KFCT 2018 (FAO and Government of Kenya), sheet
"04 USERDATABASE-KFCT2018", per 100 g edible portion. Item codes are in the `source` column.

## Amino acids - Sourced (KFCT first, USDA where KFCT has no data)
- KFCT 2018: sorghum (1039), bulrush millet (1024), kidney beans (3004), groundnut (10009).
- KFCT has no amino-acid data for whole maize flour (1022), cowpeas (3011), pigeon peas (3021)
  or soybean (3022).
- USDA FoodData Central SR Legacy (public domain), where KFCT has no amino-acid data:
  maize flour, cowpea, pigeon pea, soybean.
  KFCT itself cites USDA SR28 for many foods.
- Conversion: mg per g protein = mg amino acid per 100 g / g protein per 100 g.
  SAA = methionine + cystine.

### Cross-check KFCT vs international references (mg per g protein; full table in cleaning_report.md)
| Food | Amino acid | KFCT | USDA / Feedipedia | Verdict |
|---|---|---|---|---|
| Kidney beans | Lysine | 69.8 | 68.6 | agree |
| Kidney beans | Met+Cys | 18.0 | 25.9 | KFCT 31% lower |
| Groundnut | Lysine | 35.5 | 35.6 | agree |
| Sorghum | Lysine | 34.7 | ~22 (Feedipedia) | KFCT much higher |
| Bulrush millet | Lysine | 45.7 | ~19 (USDA proso millet) | KFCT much higher |

Sensitivity test: replacing the KFCT sorghum and millet lysine with the international values
gives the SAME top 3 blends at 15 g protein; only the scores fall (PDCAAS of blend A 0.91 -> 0.78).
The recommendation is robust to this disagreement.

## Digestibility - ASSUMED
Typical literature values, not yet cited per ingredient. Say "assumed" in the pitch.

## Prices - Sourced (FAOSTAT producer prices)
- FAOSTAT, Producer Prices domain, Kenya, Producer Price (LCU/tonne), annual values, all flagged
  "Official value". Raw file: `data/raw/FAOSTAT_producer_prices_Kenya_2020-2024.csv`.
- Price used = mean of 2022-2024 (the 3 latest years all 8 crops share), KES/tonne / 1000.
  A 3-year mean smooths big swings (e.g. pigeon peas fell from 63,708 to 36,499 KES/t in 2024).
- Groundnuts: FAOSTAT prices groundnuts in the shell; divided by an ASSUMED 70% kernel out-turn.
- These are farm-gate prices, the same basis for all 8 ingredients. Our user is a miller who buys
  grain, so they are the starting point for raw-material cost; the app lets users enter their own.
- Cross-check: REACH JMMI Q4 2025 retail prices (`data/inputs/retail_prices_reach.csv`) are
  1.4x-2.6x the FAOSTAT prices. The mark-up differs by crop (cowpeas 2.6x, beans 1.4x), so retail
  rankings can differ from farm-gate rankings. Stated as a limitation.

## Product-design constraint - ASSUMED
At farm-gate prices cowpeas are cheaper than every cereal, so without a limit the optimiser
recommends 100% cowpea, which is not a usable porridge flour. The app therefore caps the legume
share of the blend (default 40%, slider 10-100%). The default is an assumption to be set from
kitchen trials.

## Reference pattern - Sourced (verify)
FAO/WHO/UNU 2007 adult amino-acid scoring pattern (WHO Technical Report Series 935).

## Model output - Model-generated
Explanations and processing steps come from Claude on Amazon Bedrock.
