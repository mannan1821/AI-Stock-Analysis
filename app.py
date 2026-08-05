import os
from datetime import datetime
import streamlit as st
import yfinance as yf
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="centered")
st.title("📈 AI Stock Market Assistant")
st.caption("Live price lookup · Trend explanation · News search · Buy/sell analysis · Datetime-aware answers")
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
        options=["gemini-3.5-flash", "gemini-3.5-flash-lite"],
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


SYSTEM_PROMPT = (
    "You are an AI stock market assistant. "
    "You have access to tools for: (1) looking up the latest live stock price, "
    "(2) analyzing recent price trends and momentum, (3) searching recent news "
    "for a stock, (4) fetching valuation data and analyst consensus, and "
    "(5) getting the current date/time. "
    "Always ground price, trend, valuation, and time claims in tool calls "
    "rather than assumptions - never guess a price, a reason, a valuation "
    "figure, or the current date. When asked about how a stock is doing, call "
    "the trend tool and explain the direction, percentage change, and "
    "momentum in plain language. When asked WHY a stock went up or down, call "
    "the news tool and ground your explanation in the actual headlines "
    "returned - cite the headline and source rather than speculating, and say "
    "so plainly if no clear news explains the move.\n\n"
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
        tools=[get_stock_price, get_stock_trend, get_stock_news, get_stock_fundamentals, get_current_time],
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
                try:
                    label = template.format(
                        **args,
                        symbol=args.get("symbol", "the stock"),
                        period=args.get("period", "1mo"),
                    )
                except Exception:
                    label = template
                lines.append(f"- {label}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("summary"):
            with st.expander("📋 What I looked up"):
                st.markdown(msg["summary"])

prompt = st.chat_input("Ask about a stock price, trend, or the current time...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        summary = ""
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                answer = extract_text(response["messages"][-1].content)
                summary = summarize_tool_calls(response["messages"])
            except Exception as e:
                answer = f"Something went wrong: {e}"
            st.markdown(answer)
            if summary:
                with st.expander("📋 What I looked up"):
                    st.markdown(summary)

    st.session_state.messages.append({"role": "assistant", "content": answer, "summary": summary})
