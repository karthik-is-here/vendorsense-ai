# VendorSense AI

An AI **agent** (not a passive dashboard) that helps small street/market vendors
track sales via free-text input and automatically surfaces business insights —
low stock warnings, best sellers, revenue trends — after every sale, with no
button click required. Built for SDG 8: Decent Work & Economic Growth.

## Why this counts as "agentic"

Most sales trackers are: *store data -> wait for user to click "view report."*

VendorSense instead runs a closed loop on every single entry:

```
PERCEIVE  ->  PARSE (LLM)  ->  REMEMBER (SQLite)  ->  ANALYSE (pandas)  ->  ACT (alerts)
```

`app.py`'s `handle_new_sale()` triggers steps 4 and 5 automatically, immediately
after every `insert_sale()` call — see the comments in `app.py` and `agent.py`
for exactly where. There is deliberately no "Run analysis" button anywhere.

Two of the insights are genuine *inferences*, not lookups, since the system has
no stock-count field at all:
- **Low stock** is inferred from sale *frequency* (`agent.infer_low_stock`) —
  an item sold in 3+ separate transactions in the last 3 days is flagged as
  likely running low.
- **Stale items** are inferred from sale *recency* (`agent.infer_stale_items`)
  — an item that hasn't sold in 4+ days is flagged as a discount candidate.

## File structure

```
vendorsense/
├── app.py              # Streamlit UI, main entry point, the agent loop
├── agent.py             # LLM parsing (swappable provider) + analysis/insight logic
├── database.py           # SQLite schema, seed data, queries
├── requirements.txt
```

## Setup

### 1. Get a free Gemini API key

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with a Google account and click "Create API key" (free tier is
   generous enough for a hackathon demo — no billing setup required).
3. Copy the key.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Either as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"       # macOS/Linux
setx GEMINI_API_KEY "your-key-here"          # Windows
```

...or, for Streamlit Cloud deployment, add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

(`agent.py`'s `GeminiProvider` checks `os.environ["GEMINI_API_KEY"]` — Streamlit
automatically exposes `st.secrets` entries as environment variables too, but if
you use `secrets.toml` locally you may need `os.environ["GEMINI_API_KEY"] =
st.secrets["GEMINI_API_KEY"]` near the top of `app.py`.)

### 4. Run it

```bash
streamlit run app.py
```

The database (`vendorsense.db`) is created and pre-populated with ~20 realistic
sample sales on first run, so the dashboard looks populated immediately —
no need to type entries live during a demo unless you want to.

## Swapping the LLM provider

All parsing goes through the `LLMProvider` abstract base class in `agent.py`.
To use a different model, implement one new class with a `parse_sale(text) ->
dict` method (see `GeminiProvider` as a template) and point
`agent.get_default_provider()` at it. Nothing in `app.py` or `database.py`
needs to change.
