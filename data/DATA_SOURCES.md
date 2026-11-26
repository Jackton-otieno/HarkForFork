# Data sources and provenance

**How the data was cleaned:** run `python data/build_ingredients.py`. It reads the raw
`data/raw/Kenya_Food_Composition_Tables_2018.xlsx`, applies 7 documented steps, writes
`data/ingredients.csv` and a step-by-step `data/cleaning_report.md`. Inputs it uses:
`data/inputs/ingredient_map.csv`, `reference_amino_acids.csv`, `prices.csv`.

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

## Prices - Sourced / PLACEHOLDER
- Maize flour, beans, cowpeas, pigeon peas: REACH / Kenya Cash Working Group, Joint Market
  Monitoring Initiative (JMMI) Q4 2025, median retail prices across 40 markets in 7 ASAL
  counties, collected 22 Dec 2025 - 11 Jan 2026. Nairobi prices may differ.
- Sorghum flour, bulrush millet flour, groundnuts, soybeans: PLACEHOLDER. Collect at a Nairobi
  market (photo of price tag + date + market name) = MEASURED data.

## Reference pattern - Sourced (verify)
FAO/WHO/UNU 2007 adult amino-acid scoring pattern (WHO Technical Report Series 935).

## Model output - Model-generated
Explanations and processing steps come from Claude on Amazon Bedrock.
