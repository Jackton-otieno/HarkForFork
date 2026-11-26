"""LegumeLab - Streamlit demo.

Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from legumelab.optimizer import load_data, top_distinct_blends, tradeoff_curve

st.set_page_config(page_title="LegumeLab", page_icon="🌱", layout="wide")
st.title("🌱 LegumeLab")
st.caption("Find the best local legume + cereal blend for a protein flour - before the kitchen trial.")

ing, ref = load_data()

_src_cols = [c for c in ("source", "aa_source", "price_source") if c in ing.columns]
_placeholder = ing[_src_cols].apply(lambda col: col.str.upper().str.startswith("PLACEHOLDER")).any(axis=1)
if _placeholder.any():
    st.warning(
        "⚠️ Some values are still PLACEHOLDERS: "
        + ", ".join(ing.loc[_placeholder, "ingredient"])
        + ". Collect their market prices before the pitch (see data/DATA_SOURCES.md)."
    )

# ---------------- Sidebar: user goals ----------------
st.sidebar.header("Your targets")
min_protein = st.sidebar.slider("Minimum protein (g per 100 g)", 8.0, 25.0, 15.0, 0.5)
min_score = st.sidebar.slider("Minimum protein quality (amino-acid score)", 0.5, 1.0, 0.9, 0.05)
excluded = st.sidebar.multiselect("Exclude ingredients (allergy, availability)", list(ing["ingredient"]))

st.sidebar.header("Market prices (KES/kg)")
st.sidebar.caption("Change a price and the recommendation updates.")
for i, row in ing.iterrows():
    ing.at[i, "price_kes_per_kg"] = st.sidebar.number_input(
        row["ingredient"], min_value=1.0, value=float(row["price_kes_per_kg"]), step=5.0
    )

# ---------------- Recommended blends ----------------
blends = top_distinct_blends(ing, ref, min_protein, min_score, k=3, excluded=excluded)

if not blends:
    st.warning("No blend can meet these targets with the selected ingredients. "
               "Lower the protein or quality target, or allow more ingredients.")
    st.stop()

st.subheader("Top blends to test in the kitchen")
rows = [b.as_row() for b in blends]
table = pd.DataFrame(rows, index=[f"Blend {chr(65 + i)}" for i in range(len(rows))]).fillna(0)
ing_cols = [c for c in table.columns if c.endswith(" %")]
metric_cols = ["protein g/100g", "AA score", "PDCAAS", "limiting AA", "cost KES/kg"]
table = table[ing_cols + metric_cols]
st.dataframe(table, width="stretch")

cols = st.columns(len(blends))
for col, (name, b) in zip(cols, zip(table.index, blends)):
    shares = {k: v for k, v in b.fractions.items() if v > 0.001}
    fig = px.pie(names=list(shares), values=list(shares.values()), title=name, hole=0.45)
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0), showlegend=True)
    col.plotly_chart(fig, width="stretch")

# ---------------- Trade-off curve ----------------
st.subheader("Cost vs protein trade-off")
targets = [round(8 + 0.5 * i, 1) for i in range(0, 35)]
curve = tradeoff_curve(ing, ref, min_score, targets, excluded=excluded)
if not curve.empty:
    fig = px.line(curve, x="protein g/100g", y="cost KES/kg", markers=True,
                  hover_data=["AA score", "limiting AA"])
    fig.add_vline(x=min_protein, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each point is the cheapest blend that reaches that protein level "
               f"with an amino-acid score of at least {min_score}.")

# ---------------- AI explanation ----------------
st.subheader("Explain the recommendation")
choice = st.selectbox("Blend", list(table.index))
if st.button("Explain with AI"):
    from legumelab.bedrock import explain_blend
    idx = list(table.index).index(choice)
    goal = f"protein >= {min_protein} g/100g, amino-acid score >= {min_score}, lowest cost"
    with st.spinner("Asking Bedrock..."):
        try:
            text = explain_blend(rows[idx], [r for j, r in enumerate(rows) if j != idx], goal)
            st.markdown(text)
            st.caption("Model-generated explanation. Numbers come from the optimiser; "
                       "processing steps must be validated in a kitchen trial.")
        except Exception as e:  # show the error instead of crashing the demo
            st.error(f"Bedrock call failed: {e}")

with st.expander("Data provenance"):
    st.dataframe(ing[["ingredient"] + _src_cols], width="stretch")
