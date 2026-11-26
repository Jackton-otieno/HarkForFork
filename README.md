# LegumeLab – setup guide

LegumeLab recommends the cheapest blend of local legumes and cereals that reaches a protein
target and a protein-quality target, then explains it with a model on Amazon Bedrock.
Track 2 – Hack the Fork Kenya.

## Who does what (team of 4)

| Person | Job |
|---|---|
| A – Data | Replace the placeholder values in `data/ingredients.csv` with sourced values (KFCT, FAO/USDA amino acids, KAMIS prices). Fill in the `source` column. |
| B – Optimiser | `legumelab/optimizer.py`: tune the constraints, add more ingredients, test edge cases. |
| C – AWS / AI | Bedrock access, prompt, Knowledge Base (S3 + PDFs), `legumelab/bedrock.py`. |
| D – App & pitch | `app.py` (Streamlit), demo script, slides, demo video. |

---

## Step 1 – GitHub (one person, 5 min)
1. Create a **private** repo on GitHub, e.g. `legumelab`, and add your teammates.
2. Upload this starter folder to it (or `git init`, `git add .`, `git commit`, `git push`).
3. Everyone runs `git clone https://github.com/<you>/legumelab.git`.
4. Push **at least every hour**.

## Step 2 – Each laptop (10 min)
Install **Python 3.10+**, **VS Code** (with the Python extension) and **git**, then in the VS Code terminal:

```bash
cd legumelab
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Test that the optimiser runs (no AWS needed):
```bash
python -m legumelab.optimizer
streamlit run app.py
```
The app opens at http://localhost:8501. A red banner shows while the data is still placeholder.

## Step 3 – AWS access (each person, 5 min)
1. Sign in to **AWS Workshop Studio** with your email one-time password and the team's 12-character code.
2. Check the console region is **US East (N. Virginia) / us-east-1** and everyone has the same account number.
3. On the event page choose **Get AWS CLI credentials**, copy the `export ...` lines
   (or the PowerShell `$Env:...` lines on Windows) and paste them into your VS Code terminal.
4. Check it works: `aws sts get-caller-identity` (install the AWS CLI if needed),
   or `python -c "import boto3; print(boto3.client('sts').get_caller_identity())"`.

Credentials expire: when you get an "expired token" error, paste fresh ones.
**Never put them in a file in the repo.**

## Step 4 – Bedrock model (Person C, 10 min)
1. Console → **Amazon Bedrock** → **Playground** → Chat. Pick **Claude Haiku 4.5** and send "hello".
   If a form appears asking for your use case, one person fills it in once for the team.
2. Copy the model's ID: **Model catalog** → the model → **Model ID**. If the console uses a
   cross-region **inference profile** (ID starting `us.`), use that ID instead.
3. In the terminal:
   ```bash
   export LEGUMELAB_MODEL_ID="<the ID you copied>"     # Windows: $Env:LEGUMELAB_MODEL_ID="..."
   ```
4. Restart Streamlit and press **Explain with AI** in the app.

Use Haiku 4.5 or Nova Lite while building. Switch to Claude Sonnet 5 for the final demo if the
explanations need to be better. The code already waits 1 s between calls and caches answers.

## Step 5 – Knowledge Base (Person C, optional, Friday)
Grounds the processing advice in real documents, with citations.
1. Put public PDFs in `docs/knowledge_base/` (see the note there).
2. Console → **S3** → create bucket `legumelab-kb-<teamname>` in us-east-1
   (keep **Block Public Access ON**). Upload the PDFs.
3. Console → **Bedrock** → **Knowledge Bases** → Create → *Knowledge base with vector store* →
   data source = your S3 bucket → embeddings = **Titan Text Embeddings** → quick-create the vector store.
   Then press **Sync**.
4. Copy the Knowledge Base ID and set `export LEGUMELAB_KB_ID="..."`.
5. Use `ask_knowledge_base()` in `legumelab/bedrock.py` (e.g. add a "Ask the processing guide" box in the app).
6. Write down every setting you chose in this README – the account is deleted on Sunday.

## Step 6 – Real data (Person A, Thursday night → Friday morning)
See `data/DATA_SOURCES.md`. For each ingredient fill in protein, the four amino acids
(mg per g protein), digestibility and price, and write the exact source in the `source` column.
When no row says `PLACEHOLDER`, the red banner disappears.

## Step 7 – Demo hosting (optional)
Running `streamlit run app.py` on a laptop during the pitch is enough.
For a URL: launch a small **EC2** instance (t3.medium, Amazon Linux), clone the repo,
`pip install -r requirements.txt`, `streamlit run app.py --server.port 8501`, and open port 8501
in the security group to your own IP only.

## Final hours checklist
- Three hours before the end: stop adding features, push everything.
- Record a 2–3 minute demo video while everything works.
- Copy your prompts, Knowledge Base settings and data into the repo.
- Check no keys or tokens are in the repo.

## How it works (for the pitch)
- **Optimiser (the core AI):** linear programming (SciPy / HiGHS). It minimises cost per kg subject to
  protein ≥ target, and for every essential amino acid a: Σ xᵢ·pᵢ·(aaᵢ,ₐ − s·refₐ) ≥ 0,
  which guarantees an amino-acid score ≥ s. Sweeping the targets draws the cost/protein trade-off curve.
- **Language model (Bedrock):** explains the blend and suggests processing steps, using only
  the numbers the optimiser produced. Its output is labelled model-generated.
- **Limits:** predicts nutrition and cost, not taste, texture or shelf life. Food-table values are
  averages. Prices change weekly. The tool narrows the kitchen trial to 2–3 blends; it does not replace it.

## Rebuilding the data
The optimiser reads `data/ingredients.csv`. It is built from the raw Kenya Food Composition Tables:
```bash
python data/build_ingredients.py      # Windows: python data\build_ingredients.py
```
This rewrites `data/ingredients.csv` and `data/cleaning_report.md`. To change a price, edit
`data/inputs/prices.csv` and rerun; to use a different KFCT item, edit `data/inputs/ingredient_map.csv`.
