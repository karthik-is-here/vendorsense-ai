"""
app.py — Streamlit UI and orchestration for VendorSense AI.

THE AGENT LOOP lives in handle_new_sale() below:
  1. PERCEIVE  — vendor types a free-text sale entry
  2. PARSE     — agent.py's LLM provider turns it into structured data
  3. REMEMBER  — database.py stores it (the agent's long-term memory)
  4. ANALYSE   — agent.py re-reads the ENTIRE sales history and re-runs
                 every analysis function from scratch
  5. ACT       — the resulting insights are surfaced immediately, unprompted

Steps 4 and 5 happen automatically on every single new sale. There is no
"Run analysis" or "Refresh insights" button anywhere in this app — that
absence is intentional and is the whole point: it's what makes this an
agent rather than a dashboard the vendor has to operate.
"""

import streamlit as st

import agent
import database

# Load GEMINI_API_KEY from a local .env file if present (for local dev).
# No-op if python-dotenv isn't installed or .env doesn't exist — on Streamlit
# Cloud you'll use st.secrets instead (see README).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="VendorSense AI", page_icon="🛒", layout="wide")

# --- one-time setup per session: create the DB and seed it if empty, so the
# dashboard is populated for a demo even before any live entry is typed. ---
database.init_db()
database.seed_sample_data()


@st.cache_resource
def get_provider() -> agent.LLMProvider:
    # Swappable backend: change this one line to switch the whole app's LLM
    # provider (e.g. agent.OpenAIProvider()) without touching anything below.
    return agent.get_default_provider()


def handle_new_sale(raw_text: str) -> None:
    """The full agentic pipeline, triggered the instant a vendor submits an entry."""
    provider = get_provider()
    parsed = agent.parse_sale_entry(raw_text, provider=provider)

    database.insert_sale(
        item=parsed["item"],
        quantity=parsed["quantity"],
        unit=parsed["unit"],
        price=parsed["price"],
        raw_text=raw_text,
        is_estimate=parsed["is_estimate"],
    )

    if parsed["is_estimate"]:
        note = f" — {parsed['notes']}" if parsed.get("notes") else ""
        st.session_state["last_estimate_flag"] = (
            f"🔎 Logged as an estimate{note}. Check the entry below and edit if needed."
        )
    else:
        st.session_state["last_estimate_flag"] = None

    # AUTONOMOUS STEP: the agent re-reads its entire memory and re-decides
    # what's worth telling the vendor, with no user action required.
    df = database.get_sales_df()
    st.session_state["analysis"] = agent.analyze_sales(df)


# Run the analysis once on first load too, so insights are visible even
# before the vendor logs anything new in this session.
if "analysis" not in st.session_state:
    st.session_state["analysis"] = agent.analyze_sales(database.get_sales_df())
    st.session_state["last_estimate_flag"] = None

st.title("🛒 VendorSense AI")
st.caption(
    "An AI agent for street & market vendors — log sales in plain language, "
    "and it watches your business for you. (SDG 8: Decent Work & Economic Growth)"
)

# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------
with st.form("sale_entry_form", clear_on_submit=True):
    raw_text = st.text_input(
        "Log a sale — type it however you'd say it out loud",
        placeholder="e.g. sold 5kg onions for 200 rupees",
    )
    submitted = st.form_submit_button("Log sale")

if submitted and raw_text.strip():
    with st.spinner("Agent is parsing the entry and re-analysing your sales..."):
        try:
            handle_new_sale(raw_text.strip())
        except Exception as e:  # keep the app usable even if a call fails
            st.error(f"Couldn't process that entry: {e}")

if st.session_state.get("last_estimate_flag"):
    st.warning(st.session_state["last_estimate_flag"])

st.divider()

# ---------------------------------------------------------------------
# PROACTIVE INSIGHTS — the agent's autonomous output, shown prominently
# ---------------------------------------------------------------------
st.subheader("🤖 Agent Insights")
st.caption("Generated automatically after every sale — nothing here was requested by a button.")

insights = st.session_state["analysis"]["insights"]
if insights:
    for line in insights:
        st.info(line)
else:
    st.write("No insights yet — log a sale to get started.")

st.divider()

# ---------------------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------------------
trend = st.session_state["analysis"]["trend"]
if trend:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Today's revenue",
        f"₹{trend.get('today_total', 0):.0f}",
        delta=(f"{trend['day_pct_change']:.0f}%" if trend.get("day_pct_change") is not None else None),
    )
    c2.metric(
        "This week's revenue",
        f"₹{trend.get('this_week_total', 0):.0f}",
        delta=(f"{trend['week_pct_change']:.0f}%" if trend.get("week_pct_change") is not None else None),
    )
    bw = st.session_state["analysis"]["best_worst"]
    if bw:
        c3.metric("Top seller (all time)", bw["best_by_qty"], f"{bw['best_by_qty_val']:.0f} units")

st.divider()

# ---------------------------------------------------------------------
# SALES HISTORY
# ---------------------------------------------------------------------
st.subheader("📒 Sales history")
df = database.get_sales_df()
if df.empty:
    st.write("No sales logged yet.")
else:
    display_df = df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["is_estimate"] = display_df["is_estimate"].map({1: "⚠️ estimate", 0: ""})
    display_df = display_df[["timestamp", "item", "quantity", "unit", "price", "is_estimate", "raw_text"]]
    display_df.columns = ["When", "Item", "Qty", "Unit", "Price (₹)", "Flag", "Original entry"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
