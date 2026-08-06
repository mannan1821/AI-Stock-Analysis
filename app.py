import re
import os
from datetime import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="centered")
st.title("📈 AI Stock Market Assistant")
st.caption("Live price lookup · Interactive Charts · Trend explanation · News search · Buy/sell analysis")
st.warning(
    "This tool provides informational analysis only, not financial advice. "
    "It is not a licensed financial advisor. Always do your own research or "
    "consult a licensed professional before investing.",
    icon="⚠️",
)

# ----------------------------------------------------------------------------
# Charting Helper Function
# ----------------------------------------------------------------------------
def plot_stock_chart(symbol: str, period: str, chart_type: str, indicators: list):
    """
    Fetches historical stock data and generates an interactive Plotly chart
    with up to 2 technical indicators. Defaults to Line chart.
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    if df.empty:
        st.error(f"No historical data found for {symbol} over period '{period}'.")
        return

    # Check if RSI is selected (requires a secondary y-axis subplot)
    has_rsi = "RSI (14)" in indicators
    
    if has_rsi:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25]
        )
    else:
        fig = go.Figure()

    # 1. Base Price Plot (Default: Line)
    if chart_type == "Candlestick":
        chart_obj = go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name=f"{symbol} Price"
        )
    else:
        chart_obj = go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name=f"{symbol} Close Price",
            line=dict(color='#1f77b4', width=2)
        )

    if has_rsi:
        fig.add_trace(chart_obj, row=1, col=1)
    else:
        fig.add_trace(chart_obj)

    # 2. Add Indicators (Max 2 enforced via UI)
    for ind in indicators:
        if ind == "SMA 20":
            sma20 = df['Close'].rolling(window=20).mean()
            trace = go.Scatter(x=df.index, y=sma20, mode='lines', name='SMA 20', line=dict(color='orange', width=1.5))
            fig.add_trace(trace, row=1 if has_rsi else None, col=1 if has_rsi else None)
            
        elif ind == "EMA 50":
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            trace = go.Scatter(x=df.index, y=ema50, mode='lines', name='EMA 50', line=dict(color='purple', width=1.5))
            fig.add_trace(trace, row=1 if has_rsi else None, col=1 if has_rsi else None)
            
        elif ind == "Bollinger Bands":
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            trace_upper = go.Scatter(x=df.index, y=upper_band, mode='lines', name='BB Upper', line=dict(color='gray', dash='dash'))
            trace_lower = go.Scatter(x=df.index, y=lower_band, mode='lines', name='BB Lower', line=dict(color='gray', dash='dash'), fill='tonexty')
            
            fig.add_trace(trace_upper, row=1 if has_rsi else None, col=1 if has_rsi else None)
            fig.add_trace(trace_lower, row=1 if has_rsi else None, col=1 if has_rsi else None)

        elif ind == "RSI (14)" and has_rsi:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI (14)', line=dict(color='magenta')), row=2, col=1)
            # Add RSI reference thresholds
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Figure Layout Formatting
    fig.update_layout(
        title=f"📊 Stock Chart: {symbol} ({period})",
        xaxis_rangeslider_visible=False,
        height=500 if not has_rsi else 600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def extract_tickers(text: str) -> list[str]:
    """Helper to detect stock ticker symbols in user prompt."""
    tickers = re.findall(r'\b[A-Z]{2,5}(?:\.[A-Z]{2})?\b', text.upper())
    stopwords = {"I", "A", "THE", "BUY", "SELL", "WHAT", "WHY", "HOW", "IS", "ARE", "AND", "OR", "FOR"}
    return list(dict.fromkeys([t for t in tickers if t not in stopwords]))

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
        help="Get a key from https://aistudio.google.com/apikey.",
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
    (e.g. '5d', '1mo', '3mo', '6mo', '1y') and returns a plain-language summary."""
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
    """Fetches key valuation/fundamental data and Wall Street analyst consensus."""
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
    """Fetches recent news headlines for a stock ticker."""
    stock = yf.Ticker(symbol)
    try:
        news_items = stock.news or []
    except Exception as e:
        return f"Could not fetch news for {symbol}: {e}"

    if not news_items:
        return f"No recent news found for {symbol}."

    lines = []
    for item in news_items[:max_results]:
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
    "You have access to tools for looking up live stock price, analyzing trend/momentum, "
    "searching news, fetching valuation data, and checking current date/time.\n\n"
    "When answering, provide clear, objective analysis. End responses where relevant "
    "by inviting the user to explore the interactive chart options below."
)

# ----------------------------------------------------------------------------
# Agent Init
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
# Chat state & Rendering
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("summary"):
            with st.expander("📋 What I looked up"):
                st.markdown(msg["summary"])

        # Render Chart Controls & Figure for assistant messages containing detected tickers
        if msg["role"] == "assistant" and msg.get("tickers"):
            tickers = msg["tickers"]
            st.divider()
            st.subheader("📉 Dynamic Stock Chart Controls")
            
            # Interactive user options for Chart comparison
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                selected_ticker = st.selectbox(
                    "Select Ticker", 
                    tickers, 
                    key=f"ticker_{idx}"
                )
            with col2:
                selected_period = st.selectbox(
                    "Comparison Period", 
                    options=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                    index=1, 
                    key=f"period_{idx}"
                )
            with col3:
                # Default set to "Line" (index 0)
                chart_type = st.radio(
                    "Chart Type", 
                    options=["Line", "Candlestick"], 
                    index=0,
                    horizontal=True, 
                    key=f"type_{idx}"
                )

            selected_indicators = st.multiselect(
                "Plot Technical Indicators (Max 2)",
                options=["SMA 20", "EMA 50", "RSI (14)", "Bollinger Bands"],
                default=["SMA 20"],
                max_selections=2,
                key=f"ind_{idx}",
                help="Select up to 2 technical indicators to overlay on the chart."
            )

            # Draw Plotly Chart
            plot_stock_chart(selected_ticker, selected_period, chart_type, selected_indicators)

# Handle New User Input
prompt = st.chat_input("Ask about a stock price, trend, or analysis (e.g. 'Should I buy AAPL or RELIANCE.NS?')...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        summary = ""
        tickers_found = extract_tickers(prompt)
        
        with st.spinner("Analyzing stock data..."):
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

        # Render Chart controls right away if a ticker was identified
        if tickers_found:
            st.divider()
            st.subheader("📉 Dynamic Stock Chart Controls")
            
            idx = len(st.session_state.messages)
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                selected_ticker = st.selectbox(
                    "Select Ticker", 
                    tickers_found, 
                    key=f"ticker_{idx}"
                )
            with col2:
                selected_period = st.selectbox(
                    "Comparison Period", 
                    options=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                    index=1, 
                    key=f"period_{idx}"
                )
            with col3:
                # Default set to "Line" (index 0)
                chart_type = st.radio(
                    "Chart Type", 
                    options=["Line", "Candlestick"], 
                    index=0,
                    horizontal=True, 
                    key=f"type_{idx}"
                )

            selected_indicators = st.multiselect(
                "Plot Technical Indicators (Max 2)",
                options=["SMA 20", "EMA 50", "RSI (14)", "Bollinger Bands"],
                default=["SMA 20"],
                max_selections=2,
                key=f"ind_{idx}",
                help="Select up to 2 technical indicators to overlay on the chart."
            )

            plot_stock_chart(selected_ticker, selected_period, chart_type, selected_indicators)

    # Save complete assistant payload into session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "summary": summary,
        "tickers": tickers_found
    })
