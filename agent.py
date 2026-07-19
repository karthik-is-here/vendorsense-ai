"""
agent.py — the "thinking" half of VendorSense AI.

Two responsibilities live here, and together they ARE the agentic core of
this project:

1. PARSING (perceive -> structure)
   Turn a vendor's free-text sale entry into structured data using an LLM.
   This sits behind an LLMProvider interface so the underlying model
   (Gemini today, something else tomorrow) can be swapped without touching
   app.py or database.py at all.

2. ANALYSIS + INSIGHT GENERATION (observe -> decide -> act)
   After every single new sale, the agent re-reads its own memory (the full
   sales table), re-runs its analysis from scratch, and DECIDES what is
   worth telling the vendor right now — unprompted. app.py calls
   generate_insights()/analyze_sales() immediately after every insert_sale(),
   never behind a "show insights" button. That automatic re-evaluation loop
   is what makes this an agent instead of a dashboard.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# 1. LLM PROVIDER ABSTRACTION — swappable parsing backend
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Minimal interface any LLM backend must implement. Nothing else in the
    app talks to Gemini directly — everything goes through this interface,
    so swapping providers means writing one new class, not touching app.py,
    database.py, or the analysis functions below."""

    @abstractmethod
    def parse_sale(self, text: str) -> dict:
        """Return a dict with keys: item, quantity, unit, price, is_estimate, notes."""
        raise NotImplementedError


PARSE_PROMPT = """You are a data-entry assistant for a small street/market vendor in India.
The vendor will describe a sale in free, informal text. Extract structured fields.

Rules:
- "item": the product name, lowercase, singular where natural (e.g. "onion" not "Onions").
- "quantity": a number. If a unit like kg/pieces/dozen is mentioned, use that number.
  If NO quantity is given at all, make a reasonable small-vendor estimate (e.g. 1) and set is_estimate=true.
- "unit": one of "kg", "g", "pieces", "dozen", "litre", or your best guess. Default "kg" for produce.
- "price": the TOTAL amount in rupees the vendor received for this sale (not a per-kg rate),
  as a plain number. If ambiguous, make your best reasonable estimate and set is_estimate=true.
- "is_estimate": true if you had to guess or assume ANY field because the input was vague or incomplete.
- "notes": a short (under 12 words) note explaining any assumption made, or "" if nothing was assumed.

Respond with ONLY a JSON object, no markdown code fences, no extra commentary, in exactly this shape:
{"item": "...", "quantity": 0, "unit": "...", "price": 0, "is_estimate": false, "notes": "..."}

Vendor's entry: "{TEXT}"
"""


class GeminiProvider(LLMProvider):
    """Gemini-backed implementation using the official google-generativeai SDK."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        import google.generativeai as genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY as an environment "
                "variable, or in Streamlit secrets (see README for setup)."
            )
        # transport="rest" forces plain HTTPS instead of gRPC. The default gRPC
        # transport loads a native cygrpc DLL that locked-down Windows machines
        # (college/work laptops with Application Control policies) often block
        # outright — REST avoids that dependency entirely and is plenty fast
        # for single-request calls like this.
        genai.configure(api_key=api_key, transport="rest")
        self._model = genai.GenerativeModel(model_name)

    def parse_sale(self, text: str) -> dict:
        prompt = PARSE_PROMPT.replace("{TEXT}", text)
        response = self._model.generate_content(prompt)
        raw = (response.text or "").strip()
        # Gemini sometimes wraps JSON in ```json fences despite instructions -- strip them.
        raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fail safe rather than crashing the agent loop: log a clearly-flagged
            # placeholder entry so the vendor can see something went wrong and fix it.
            return {
                "item": "unknown",
                "quantity": 1.0,
                "unit": "kg",
                "price": 0.0,
                "is_estimate": True,
                "notes": "could not parse model output, please edit",
            }

        return _normalize_parsed(data)


def _normalize_parsed(data: dict) -> dict:
    """Defensive type coercion so a slightly malformed model response never
    crashes the app — bad/missing fields fall back to safe defaults and get
    flagged as estimates rather than silently corrupting the database."""
    item = str(data.get("item", "unknown")).strip().lower() or "unknown"
    unit = str(data.get("unit", "kg")).strip().lower() or "kg"

    try:
        quantity = float(data.get("quantity", 1) or 1)
    except (TypeError, ValueError):
        quantity = 1.0

    try:
        price = float(data.get("price", 0) or 0)
    except (TypeError, ValueError):
        price = 0.0

    return {
        "item": item,
        "quantity": quantity,
        "unit": unit,
        "price": price,
        "is_estimate": bool(data.get("is_estimate", False)),
        "notes": str(data.get("notes", "")),
    }


def get_default_provider() -> LLMProvider:
    """Factory function — change this one line to switch the entire app's
    LLM backend (e.g. to a future OpenAIProvider or LocalModelProvider)."""
    return GeminiProvider()


def parse_sale_entry(text: str, provider: LLMProvider | None = None) -> dict:
    """Convenience wrapper used by app.py."""
    provider = provider or get_default_provider()
    return provider.parse_sale(text)


# ---------------------------------------------------------------------------
# 2. AUTONOMOUS ANALYSIS + INSIGHT GENERATION
# ---------------------------------------------------------------------------


def _last_n_days(df: pd.DataFrame, n: int, ref: datetime | None = None) -> pd.DataFrame:
    ref = ref or datetime.now()
    cutoff = ref - timedelta(days=n)
    return df[df["timestamp"] >= cutoff]


def best_worst_items(df: pd.DataFrame) -> dict:
    """Best/worst sellers by both quantity and revenue, over all-time history."""
    if df.empty:
        return {}
    by_qty = df.groupby("item")["quantity"].sum().sort_values(ascending=False)
    by_rev = df.groupby("item")["price"].sum().sort_values(ascending=False)
    return {
        "best_by_qty": by_qty.index[0],
        "best_by_qty_val": float(by_qty.iloc[0]),
        "worst_by_qty": by_qty.index[-1],
        "worst_by_qty_val": float(by_qty.iloc[-1]),
        "best_by_rev": by_rev.index[0],
        "best_by_rev_val": float(by_rev.iloc[0]),
        "worst_by_rev": by_rev.index[-1],
        "worst_by_rev_val": float(by_rev.iloc[-1]),
    }


def infer_low_stock(df: pd.DataFrame, window_days: int = 3, min_occurrences: int = 3) -> list[str]:
    """
    AGENTIC INFERENCE — there is no stock-count field anywhere in this system.
    The agent *infers* "probably running low" purely from sale frequency: if
    an item has been sold in at least `min_occurrences` separate transactions
    within the last `window_days`, it's moving fast enough that the vendor
    should probably restock soon. This is a judgment call the agent makes on
    its own from behavioral data, not a lookup.
    """
    recent = _last_n_days(df, window_days)
    if recent.empty:
        return []
    counts = recent.groupby("item").size()
    return counts[counts >= min_occurrences].index.tolist()


def infer_stale_items(df: pd.DataFrame, stale_days: int = 4) -> list[dict]:
    """The flip side of low-stock inference: items that exist in sales history
    but haven't sold in a while. Flagged as candidates for a discount/promo."""
    if df.empty:
        return []
    last_sold = df.groupby("item")["timestamp"].max()
    now = datetime.now()
    stale = []
    for item, last_ts in last_sold.items():
        days_since = (now - last_ts).days
        if days_since >= stale_days:
            stale.append({"item": item, "days_since": days_since})
    return sorted(stale, key=lambda x: -x["days_since"])


def revenue_trend(df: pd.DataFrame) -> dict:
    """Today vs. yesterday, and this week vs. last week."""
    if df.empty:
        return {}
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)

    def day_total(d):
        mask = df["timestamp"].dt.date == d
        return float(df.loc[mask, "price"].sum())

    today_total = day_total(today)
    yesterday_total = day_total(yesterday)
    day_pct_change = None
    if yesterday_total > 0:
        day_pct_change = ((today_total - yesterday_total) / yesterday_total) * 100

    this_week_total = float(_last_n_days(df, 7, now)["price"].sum())
    last_week_mask = (df["timestamp"] < now - timedelta(days=7)) & (
        df["timestamp"] >= now - timedelta(days=14)
    )
    last_week_total = float(df.loc[last_week_mask, "price"].sum())
    week_pct_change = None
    if last_week_total > 0:
        week_pct_change = ((this_week_total - last_week_total) / last_week_total) * 100

    return {
        "today_total": today_total,
        "yesterday_total": yesterday_total,
        "day_pct_change": day_pct_change,
        "this_week_total": this_week_total,
        "last_week_total": last_week_total,
        "week_pct_change": week_pct_change,
    }


def generate_insights(df: pd.DataFrame) -> list[str]:
    """
    THE AGENTIC CORE OF VENDORSENSE.

    app.py calls this automatically every time a new sale is logged — never
    behind a button. It re-reads the full sales history, re-runs every
    analysis function above, and DECIDES which findings are worth surfacing
    to the vendor right now, in priority order. That "notice things and
    decide what to say" step — happening on its own, after every write — is
    what separates an agent from a passive reporting dashboard.
    """
    if df.empty:
        return ["No sales logged yet. Log your first sale to get started!"]

    insights: list[str] = []

    # -- Low stock inference (highest priority: actionable, time-sensitive) --
    for item in infer_low_stock(df):
        insights.append(
            f"⚠️ Low stock likely: {item} — sold frequently in the last 3 days, consider restocking"
        )

    # -- Best seller today --
    today_df = df[df["timestamp"].dt.date == datetime.now().date()]
    if not today_df.empty:
        best_today = today_df.groupby("item")["quantity"].sum().idxmax()
        insights.append(f"📈 Best seller today: {best_today}")

    # -- Revenue trend --
    trend = revenue_trend(df)
    if trend:
        today_total = trend["today_total"]
        if trend["day_pct_change"] is not None:
            direction = "up" if trend["day_pct_change"] >= 0 else "down"
            insights.append(
                f"💰 Today's earnings: ₹{today_total:.0f}, {direction} "
                f"{abs(trend['day_pct_change']):.0f}% from yesterday"
            )
        elif today_total > 0:
            insights.append(f"💰 Today's earnings so far: ₹{today_total:.0f}")

        if trend["week_pct_change"] is not None:
            direction = "up" if trend["week_pct_change"] >= 0 else "down"
            insights.append(
                f"📊 This week's revenue is {direction} {abs(trend['week_pct_change']):.0f}% "
                f"vs last week (₹{trend['this_week_total']:.0f} vs ₹{trend['last_week_total']:.0f})"
            )

    # -- Stale items (discount candidates) --
    for s in infer_stale_items(df):
        insights.append(
            f"🔻 {s['item'].title()} hasn't sold in {s['days_since']} days, consider a discount"
        )

    # -- Overall top seller (lower priority context) --
    bw = best_worst_items(df)
    if bw:
        insights.append(
            f"🏆 Top seller overall: {bw['best_by_qty']} ({bw['best_by_qty_val']:.0f} units sold)"
        )

    return insights


def analyze_sales(df: pd.DataFrame) -> dict:
    """Bundles every analysis function into one result, including the final
    proactive insights list. This is the single call app.py makes right
    after every insert_sale() to re-run the agent's full observe-decide cycle."""
    return {
        "best_worst": best_worst_items(df),
        "low_stock": infer_low_stock(df),
        "stale_items": infer_stale_items(df),
        "trend": revenue_trend(df),
        "insights": generate_insights(df),
    }
