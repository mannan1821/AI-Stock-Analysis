import os
from datetime import datetime

import streamlit as st
import yfinance as yf
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="centered")
st.title("📈 AI Stock Market Assistant")
st.caption("Live price lookup · Trend explanation · Datetime-aware answers")

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
        options=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
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
def get_current_time() -> str:
    """Returns the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


SYSTEM_PROMPT = (
    "You are an AI stock market assistant. "
    "You have access to tools for: (1) looking up the latest live stock price, "
    "(2) analyzing recent price trends and momentum, and (3) getting the current date/time. "
    "Always ground price and time claims in tool calls rather than assumptions - "
    "never guess a price or the current date. When asked about how a stock is doing, "
    "call the trend tool and explain the direction, percentage change, and momentum in "
    "plain language. When a question depends on 'today', 'now', or recency, call the "
    "current-time tool first so your answer is date/time-aware. If a bare ticker like "
    "'RELIANCE' fails, try common suffixes such as '.NS' (NSE) or '.BO' (BSE)."
)


# ----------------------------------------------------------------------------
# Agent (cached so it isn't rebuilt on every rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent(api_key: str, model_name: str):
    model = ChatGoogleGenerativeAI(model=model_name)
    return create_agent(
        model=model,
        tools=[get_stock_price, get_stock_trend, get_current_time],
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


# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a stock price, trend, or the current time...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                answer = extract_text(response["messages"][-1].content)
            except Exception as e:
                answer = f"Something went wrong: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
