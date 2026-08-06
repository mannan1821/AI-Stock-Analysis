import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="centered")
st.title("📈 AI Stock Market Assistant")
st.caption("Live price lookup · Trend explanation · News search · Interactive charts · Buy/sell analysis · Datetime-aware answers")
st.warning(
    "This tool provides informational analysis only, not financial advice. "
    "It is not a licensed financial advisor. Always do your own research or "
    "consult a licensed professional before investing.",
    icon="⚠️",
)

# ----------------------------------------------------------------------------
# API key handling
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    default_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google API Key",
        value=default_key,
        type="password",
        help="Get a key from https://aistudio.google.com/apikey. "
        "You can also set the GOOGLE_API_KEY environment variable instead.",
    )
    model_name = st.selectbox(
        "Model",
        options=["gemini-3.5-flash-lite", "gemini-3.5-flash"],
        index=0,
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("Enter your Google API Key in the sidebar to get started.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
@tool
def get_stock_price(symbol: str) -> float:
    """Returns the latest closing stock price for a given ticker symbol
    (e.g. 'RELIANCE.NS' for NSE, 'AAPL' for US markets)."""
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")
    if price.empty:
        raise ValueError(f"No price data found for symbol '{symbol}'.")
    return float(price["Close"].iloc[-1])


@tool
def get_stock_trend(symbol: str, period: str = "1mo") -> str:
    """Analyzes the recent price trend for a stock ticker over a given period
    (e.g. '5d', '1mo', '3mo', '6mo', '1y') and returns a plain-language summary
    covering direction, percentage change, period high/low, and short-vs-long
    moving-average momentum."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    if hist.empty:
        return f"No historical data found for {symbol} over period '{period}'."

    closes = hist["Close"]
    start_price = float(closes.iloc[0])
    end_price = float(closes.iloc[-1])
    pct_change = ((end_price - start_price) / start_price) * 100
    period_high = float(closes.max())
    period_low = float(closes.min())

    direction = "upward" if pct_change > 0 else ("downward" if pct_change < 0 else "flat")

    short_window = min(5, len(closes))
    long_window = min(20, len(closes))
    sma_short = float(closes.rolling(window=short_window).mean().iloc[-1])
    sma_long = float(closes.rolling(window=long_window).mean().iloc[-1])
    momentum = (
        "bullish (short-term average is above the long-term average)"
        if sma_short > sma_long
        else "bearish (short-term average is below the long-term average)"
    )

    return (
        f"{symbol} moved {direction} over the last {period}, changing "
        f"{pct_change:.2f}% from {start_price:.2f} to {end_price:.2f}. "
        f"Period high: {period_high:.2f}, period low: {period_low:.2f}. "
        f"Momentum currently looks {momentum}."
    )


@tool
def get_stock_fundamentals(symbol: str) -> str:
    """Fetches key valuation/fundamental data and Wall Street analyst consensus
    for a stock ticker. Use this when the user asks whether they should buy a
    stock, or wants an investment opinion/recommendation - it grounds that
    analysis in real valuation metrics and analyst ratings instead of guesses."""
    stock = yf.Ticker(symbol)
    try:
        info = stock.info or {}
    except Exception as e:
        return f"Could not fetch fundamentals for {symbol}: {e}"

    if not info:
        return f"No fundamental data found for {symbol}."

    def fmt(key, label, prefix="", suffix="", pct=False):
        val = info.get(key)
        if val is None:
            return None
        if pct:
            val = f"{val * 100:.2f}%"
        elif isinstance(val, (int, float)):
            val = f"{prefix}{val:,.2f}{suffix}"
        return f"{label}: {val}"

    lines = [x for x in [
        fmt("currentPrice", "Current price") or fmt("regularMarketPrice", "Current price"),
        fmt("trailingPE", "Trailing P/E"),
        fmt("forwardPE", "Forward P/E"),
        fmt("priceToBook", "Price-to-book"),
        fmt("marketCap", "Market cap", prefix="₹" if symbol.upper().endswith((".NS", ".BO")) else "$"),
        fmt("fiftyTwoWeekHigh", "52-week high"),
        fmt("fiftyTwoWeekLow", "52-week low"),
        fmt("dividendYield", "Dividend yield", pct=True),
        fmt("beta", "Beta (volatility vs market)"),
        fmt("recommendationKey", "Analyst consensus"),
        fmt("recommendationMean", "Analyst rating (1=Strong Buy, 5=Sell)"),
        fmt("targetMeanPrice", "Analyst avg. price target"),
        fmt("targetHighPrice", "Analyst high price target"),
        fmt("targetLowPrice", "Analyst low price target"),
        fmt("numberOfAnalystOpinions", "Number of analysts covering"),
    ] if x]

    if not lines:
        return f"No usable fundamental data found for {symbol}."

    return "\n".join(f"- {line}" for line in lines)


@tool
def get_stock_news(symbol: str, max_results: int = 5) -> str:
    """Fetches recent news headlines for a stock ticker. Use this whenever the
    user asks WHY a stock has moved, or wants the reason/context behind a
    price change - it returns real headlines, sources, and links that can
    explain recent price action."""
    stock = yf.Ticker(symbol)
    try:
        news_items = stock.news or []
    except Exception as e:
        return f"Could not fetch news for {symbol}: {e}"

    if not news_items:
        return f"No recent news found for {symbol}."

    lines = []
    for item in news_items[:max_results]:
        # yfinance's news payload shape has changed across versions - some
        # return fields at the top level, newer ones nest them under "content".
        content = item.get("content", item) if isinstance(item, dict) else {}

        title = content.get("title") or item.get("title") or "Untitled"

        provider = content.get("provider")
        publisher = (
            provider.get("displayName")
            if isinstance(provider, dict)
            else item.get("publisher")
        ) or "Unknown source"

        canonical = content.get("canonicalUrl")
        link = (
            canonical.get("url")
            if isinstance(canonical, dict)
            else item.get("link")
        ) or ""

        pub_date = content.get("pubDate") or item.get("providerPublishTime") or ""

        line = f'- "{title}" — {publisher}'
        if pub_date:
            line += f" ({pub_date})"
        if link:
            line += f" [{link}]"
        lines.append(line)

    return "\n".join(lines)


@tool
def get_current_time() -> str:
    """Returns the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------
# Charting (Plotly) — historical price charts, comparisons, and indicators
# ----------------------------------------------------------------------------
VALID_INDICATORS = {"sma20", "sma50", "ema20", "rsi", "macd", "bollinger"}

CHART_COLORS = {
    "price_up": "#1E8E5A",
    "price_down": "#B23A48",
    "SMA 20": "#C9992B",
    "SMA 50": "#0E8C8C",
    "EMA 20": "#8FD9C4",
    "BB Band": "#7C90B8",
    "RSI": "#C9992B",
    "MACD": "#0E8C8C",
    "MACD Signal": "#C9992B",
    "compare_a": "#0E8C8C",
    "compare_b": "#C9992B",
}


def _compute_indicators(hist: pd.DataFrame, indicators: List[str]) -> dict:
    """Computes the requested technical indicators from OHLC history."""
    close = hist["Close"]
    out = {}

    if "sma20" in indicators:
        out["SMA 20"] = close.rolling(window=20).mean()
    if "sma50" in indicators:
        out["SMA 50"] = close.rolling(window=50).mean()
    if "ema20" in indicators:
        out["EMA 20"] = close.ewm(span=20, adjust=False).mean()

    if "bollinger" in indicators:
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        out["BB Upper"] = sma20 + 2 * std20
        out["BB Lower"] = sma20 - 2 * std20

    if "rsi" in indicators:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss
        out["RSI"] = 100 - (100 / (1 + rs))

    if "macd" in indicators:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        out["MACD"] = macd_line
        out["MACD Signal"] = macd_line.ewm(span=9, adjust=False).mean()

    return out


def _yahoo_link(symbol: str) -> tuple:
    """Returns a (label, url) pair linking to the Yahoo Finance quote page."""
    return (symbol, f"https://finance.yahoo.com/quote/{symbol}")


def _build_chart(symbol: str, period: str, compare_symbol: Optional[str], indicators: List[str]):
    """Builds a Plotly figure for either a single-stock candlestick chart with
    optional indicator overlays, or a normalized comparison line chart of two
    tickers. Returns (figure, summary_text, links) or (None, error_text, [])."""
    hist = yf.Ticker(symbol).history(period=period)
    if hist.empty:
        return None, f"No historical data found for '{symbol}' over period '{period}'.", []

    # --- comparison mode: normalize both to a base of 100 so they're comparable ---
    if compare_symbol:
        hist2 = yf.Ticker(compare_symbol).history(period=period)
        if hist2.empty:
            return None, f"No historical data found for '{compare_symbol}' over period '{period}'.", []

        norm1 = hist["Close"] / hist["Close"].iloc[0] * 100
        norm2 = hist2["Close"] / hist2["Close"].iloc[0] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=norm1, name=symbol,
                                  line=dict(color=CHART_COLORS["compare_a"], width=2)))
        fig.add_trace(go.Scatter(x=hist2.index, y=norm2, name=compare_symbol,
                                  line=dict(color=CHART_COLORS["compare_b"], width=2)))
        fig.update_layout(
            title=f"{symbol} vs {compare_symbol} — normalized performance ({period})",
            yaxis_title="Growth (start = 100)",
            template="plotly_dark", height=450, margin=dict(t=60, b=30, l=40, r=20),
            legend=dict(orientation="h", y=1.08),
        )
        pct1 = norm1.iloc[-1] - 100
        pct2 = norm2.iloc[-1] - 100
        summary = (
            f"Comparing {symbol} ({pct1:+.2f}%) vs {compare_symbol} ({pct2:+.2f}%) "
            f"over the last {period}, both normalized to a starting value of 100."
        )
        links = [_yahoo_link(symbol), _yahoo_link(compare_symbol)]
        return fig, summary, links

    # --- single-stock candlestick chart with optional indicator overlays ---
    indicators = [i for i in (indicators or []) if i in VALID_INDICATORS]
    ind_data = _compute_indicators(hist, indicators)

    show_rsi = "rsi" in indicators
    show_macd = "macd" in indicators
    rows = 1 + show_rsi + show_macd
    row_heights = [0.55] + [0.225] * (rows - 1) if rows > 1 else [1.0]
    titles = ["Price"] + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=row_heights, subplot_titles=titles,
    )

    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=symbol,
            increasing_line_color=CHART_COLORS["price_up"], decreasing_line_color=CHART_COLORS["price_down"],
        ),
        row=1, col=1,
    )

    for name in ["SMA 20", "SMA 50", "EMA 20"]:
        if name in ind_data:
            fig.add_trace(
                go.Scatter(x=hist.index, y=ind_data[name], name=name,
                           line=dict(width=1.4, color=CHART_COLORS.get(name))),
                row=1, col=1,
            )
    if "BB Upper" in ind_data:
        fig.add_trace(go.Scatter(x=hist.index, y=ind_data["BB Upper"], name="Bollinger Upper",
                                  line=dict(width=1, color=CHART_COLORS["BB Band"], dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=ind_data["BB Lower"], name="Bollinger Lower",
                                  line=dict(width=1, color=CHART_COLORS["BB Band"], dash="dot")), row=1, col=1)

    r = 2
    if show_rsi:
        fig.add_trace(go.Scatter(x=hist.index, y=ind_data["RSI"], name="RSI",
                                  line=dict(color=CHART_COLORS["RSI"])), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=CHART_COLORS["price_down"], row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=CHART_COLORS["price_up"], row=r, col=1)
        r += 1
    if show_macd:
        fig.add_trace(go.Scatter(x=hist.index, y=ind_data["MACD"], name="MACD",
                                  line=dict(color=CHART_COLORS["MACD"])), row=r, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=ind_data["MACD Signal"], name="Signal",
                                  line=dict(color=CHART_COLORS["MACD Signal"])), row=r, col=1)

    fig.update_layout(
        title=f"{symbol} — {period} chart" + (f" with {', '.join(indicators)}" if indicators else ""),
        template="plotly_dark", height=280 * rows + 120,
        xaxis_rangeslider_visible=False, margin=dict(t=60, b=30, l=40, r=20),
        legend=dict(orientation="h", y=1.06),
    )

    summary = f"Showing the {period} price chart for {symbol}"
    if indicators:
        summary += f" with {', '.join(indicators)} plotted on it."
    else:
        summary += (
            ". Would you like me to add any indicators to it — for example a "
            "20/50-day SMA, 20-day EMA, Bollinger Bands, RSI, or MACD?"
        )
    links = [_yahoo_link(symbol)]
    return fig, summary, links


@tool
def show_stock_chart(
    symbol: str,
    period: str = "6mo",
    compare_symbol: Optional[str] = None,
    indicators: Optional[List[str]] = None,
) -> str:
    """Displays an interactive historical price chart for a stock directly in
    the chat UI, using real Yahoo Finance data. Use this whenever the user
    asks to see, show, plot, or visualize a chart for a stock.

    To compare two stocks' performance visually, pass a second ticker as
    compare_symbol - both will be normalized to a starting value of 100 so
    their growth is directly comparable regardless of price scale.

    To overlay technical indicators on a single-stock chart, pass a list of
    any of: "sma20", "sma50", "ema20", "bollinger", "rsi", "macd".

    If the user hasn't specified indicators and isn't comparing two stocks,
    after showing the plain chart, ask them in your reply whether they'd
    like indicators added - and if they say yes or name one, call this tool
    again for the same symbol/period with the indicators list filled in."""
    # Note: this only returns the text summary. The actual Plotly figure is
    # rebuilt in the main Streamlit thread after the agent finishes (see
    # `_extract_chart_calls` below) - LangChain/LangGraph can run tool calls
    # off the main thread, and Streamlit's session_state writes from a
    # background thread can silently fail to reach the UI, so we don't
    # depend on a session_state write happening inside this function.
    _, summary, _ = _build_chart(symbol, period, compare_symbol, indicators)
    return summary


def _extract_chart_calls(messages) -> list:
    """Scans the agent's response messages for show_stock_chart tool calls
    and returns their argument dicts, in the order they were called."""
    calls = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.get("name") == "show_stock_chart":
                    calls.append(tc.get("args", {}) or {})
    return calls


SYSTEM_PROMPT = (
    "You are an AI stock market assistant. "
    "You have access to tools for: (1) looking up the latest live stock price, "
    "(2) analyzing recent price trends and momentum, (3) searching recent news "
    "for a stock, (4) fetching valuation data and analyst consensus, "
    "(5) displaying an interactive historical price chart (optionally "
    "comparing two stocks, or overlaying technical indicators), and "
    "(6) getting the current date/time. "
    "Always ground price, trend, valuation, and time claims in tool calls "
    "rather than assumptions - never guess a price, a reason, a valuation "
    "figure, or the current date. When asked about how a stock is doing, call "
    "the trend tool and explain the direction, percentage change, and "
    "momentum in plain language. When asked WHY a stock went up or down, call "
    "the news tool and ground your explanation in the actual headlines "
    "returned - cite the headline and source rather than speculating, and say "
    "so plainly if no clear news explains the move.\n\n"
    "When asked to see, show, plot, visualize, or compare a stock's chart, "
    "call show_stock_chart. For a plain chart with no indicators requested "
    "and no comparison, ask afterwards whether the user wants indicators "
    "added (SMA, EMA, Bollinger Bands, RSI, or MACD). If they say yes or "
    "name specific indicators in a later message, call show_stock_chart "
    "again for the same symbol and period with the indicators list filled "
    "in - do not just describe the indicator in words, actually call the "
    "tool so it renders on the chart. To compare two stocks visually, pass "
    "both tickers to show_stock_chart via symbol and compare_symbol.\n\n"
    "When asked something like 'should I buy this stock', 'is this a good "
    "investment', or for a buy/sell/hold recommendation: call the price, "
    "trend, news, and fundamentals tools together, then present a balanced "
    "analysis with a clear 'Reasons to consider buying' list and a 'Reasons "
    "for caution' list, covering valuation (P/E, price targets), momentum, "
    "and any relevant news. If analyst consensus/price targets are "
    "available, report them factually as Wall Street's view, not your own "
    "recommendation. Do NOT issue your own definitive buy/sell verdict or "
    "tell the user what to do with their money - you are not a licensed "
    "financial advisor and this is not financial advice. Always end this "
    "kind of answer with a brief reminder that this is informational only "
    "and the user should do their own research or consult a licensed "
    "financial advisor before investing.\n\n"
    "When a question depends on 'today', 'now', or recency, call the "
    "current-time tool first so your answer is date/time-aware. If a bare "
    "ticker like 'RELIANCE' fails, try common suffixes such as '.NS' (NSE) "
    "or '.BO' (BSE)."
)


# ----------------------------------------------------------------------------
# Agent (cached so it isn't rebuilt on every rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent(api_key: str, model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name)
    return create_agent(
        model=model,
        tools=[
            get_stock_price, get_stock_trend, get_stock_news,
            get_stock_fundamentals, show_stock_chart, get_current_time,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


agent = get_agent(api_key, model_name)


def extract_text(content) -> str:
    """Normalize a LangChain message's `.content` into a plain string.

    `content` can be a plain string, or a list of content blocks (dicts)
    such as [{'type': 'text', 'text': '...'}, ...] depending on the model
    provider. This pulls out and joins just the text parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


TOOL_LABELS = {
    "get_stock_price": "Looked up the live price for {symbol}",
    "get_stock_trend": "Analyzed the {period} trend for {symbol}",
    "get_stock_news": "Searched recent news for {symbol}",
    "get_stock_fundamentals": "Checked valuation & analyst data for {symbol}",
    "show_stock_chart": "Plotted a chart for {symbol}",
    "get_current_time": "Checked the current date/time",
}


def summarize_tool_calls(messages) -> str:
    """Build a short bullet-point recap of which tools the agent actually
    called while answering, so the user can see what info backs the answer."""
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                template = TOOL_LABELS.get(name, f"Called `{name}`")
                fmt_args = dict(args)
                fmt_args.setdefault("symbol", "the stock")
                fmt_args.setdefault("period", "1mo")
                try:
                    label = template.format(**fmt_args)
                except Exception:
                    label = template

                if name == "show_stock_chart":
                    extras = []
                    if fmt_args.get("compare_symbol"):
                        extras.append(f"vs {fmt_args['compare_symbol']}")
                    if fmt_args.get("indicators"):
                        extras.append(f"with {', '.join(fmt_args['indicators'])}")
                    if extras:
                        label += " (" + ", ".join(extras) + ")"

                lines.append(f"- {label}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart") is not None:
            st.plotly_chart(msg["chart"]["fig"], use_container_width=True, key=f"chart_history_{i}")
            links = msg["chart"].get("links") or []
            if links:
                st.caption(
                    " · ".join(f"🔗 [{label} on Yahoo Finance]({url})" for label, url in links)
                )
        if msg.get("summary"):
            with st.expander("📋 What I looked up"):
                st.markdown(msg["summary"])

prompt = st.chat_input("Ask about a stock price, trend, chart, or the current time...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        summary = ""
        chart = None
        with st.spinner("Thinking..."):
            try:
                # Send the full conversation so far (not just the latest message)
                # so follow-ups like "yes, add RSI to that" retain context of
                # which symbol/period the user is referring to.
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                response = agent.invoke({"messages": history})
                answer = extract_text(response["messages"][-1].content)
                summary = summarize_tool_calls(response["messages"])

                # Rebuild the chart (if any) in this main thread from the
                # actual tool call args, rather than trusting a session_state
                # write made from inside the tool - see note in
                # show_stock_chart for why.
                chart_calls = _extract_chart_calls(response["messages"])
                if chart_calls:
                    last_call = chart_calls[-1]
                    chart_symbol = last_call.get("symbol")
                    if chart_symbol:
                        fig, _, links = _build_chart(
                            symbol=chart_symbol,
                            period=last_call.get("period", "6mo"),
                            compare_symbol=last_call.get("compare_symbol"),
                            indicators=last_call.get("indicators"),
                        )
                        if fig is not None:
                            chart = {"fig": fig, "links": links}
            except Exception as e:
                answer = f"Something went wrong: {e}"
            st.markdown(answer)
            if chart is not None:
                st.plotly_chart(chart["fig"], use_container_width=True, key="chart_new")
                links = chart.get("links") or []
                if links:
                    st.caption(
                        " · ".join(f"🔗 [{label} on Yahoo Finance]({url})" for label, url in links)
                    )
            if summary:
                with st.expander("📋 What I looked up"):
                    st.markdown(summary)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "summary": summary, "chart": chart}
    )
