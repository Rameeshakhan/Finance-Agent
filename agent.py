"""
Finance Assistant Agent — LangChain + LangGraph
Tools: Stock Price, Crypto Price, Exchange Rate, Calculator, Web Search (Tavily)
Memory: LangGraph MemorySaver (thread-based checkpointing)
"""

import os
import json
import math
import requests
from typing import Annotated, TypedDict
from datetime import datetime

# ── LangChain / LangGraph ──────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch the latest stock price for a given ticker symbol (e.g. AAPL, TSLA, MSFT).
    Uses Yahoo Finance unofficial quote API — no key required.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice", "N/A")
        prev  = meta.get("previousClose", "N/A")
        name  = meta.get("longName", symbol.upper())
        currency = meta.get("currency", "USD")
        change = round(price - prev, 2) if isinstance(price, float) and isinstance(prev, float) else "N/A"
        pct    = round((change / prev) * 100, 2) if isinstance(change, float) and prev else "N/A"
        return (
            f"📈 {name} ({symbol.upper()})\n"
            f"   Price     : {price} {currency}\n"
            f"   Prev Close: {prev} {currency}\n"
            f"   Change    : {change} ({pct}%)\n"
            f"   As of     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
    except Exception as e:
        return f"Error fetching stock data for {symbol}: {e}"


@tool
def get_crypto_price(coin_id: str) -> str:
    """
    Fetch the current price (USD) of a cryptocurrency using CoinGecko API (free, no key).
    Pass the CoinGecko coin ID, e.g. 'bitcoin', 'ethereum', 'solana', 'dogecoin'.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id.lower(),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if coin_id.lower() not in data:
            return f"Coin '{coin_id}' not found. Try 'bitcoin', 'ethereum', 'solana', etc."
        d = data[coin_id.lower()]
        price  = d.get("usd", "N/A")
        change = d.get("usd_24h_change", "N/A")
        mcap   = d.get("usd_market_cap", "N/A")
        change_str = f"{round(change, 2)}%" if isinstance(change, float) else "N/A"
        mcap_str   = f"${mcap:,.0f}" if isinstance(mcap, float) else "N/A"
        return (
            f"🪙 {coin_id.capitalize()}\n"
            f"   Price     : ${price:,.4f}\n"
            f"   24h Change: {change_str}\n"
            f"   Market Cap: {mcap_str}\n"
            f"   As of     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
    except Exception as e:
        return f"Error fetching crypto data for {coin_id}: {e}"


@tool
def get_exchange_rate(base: str, target: str) -> str:
    """
    Get the current exchange rate between two currencies.
    Uses exchangerate.host (free API, no key required).
    Example: base='USD', target='EUR'  or  base='PKR', target='USD'
    """
    try:
        url = f"https://open.er-api.com/v6/latest/{base.upper()}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("result") != "success":
            return f"Could not retrieve rates for {base.upper()}."
        rates = data.get("rates", {})
        rate  = rates.get(target.upper())
        if rate is None:
            return f"Target currency '{target}' not found."
        return (
            f"💱 Exchange Rate\n"
            f"   1 {base.upper()} = {rate:.4f} {target.upper()}\n"
            f"   Updated: {data.get('time_last_update_utc', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching exchange rate: {e}"


@tool
def finance_calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical or financial expression.
    Supports: +, -, *, /, **, sqrt(), log(), round(), abs(), etc.
    Example: '1500 * 1.07 ** 10'  or  'sqrt(144) + 200 / 4'
    """
    try:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names["abs"] = abs
        allowed_names["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"🧮 Result: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL — Tavily web search (if TAVILY_API_KEY is set)
# ══════════════════════════════════════════════════════════════════════════════

def _build_tools():
    base_tools = [get_stock_price, get_crypto_price, get_exchange_rate, finance_calculator]
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
            search.description = (
                "Search the web for latest financial news, company information, "
                "market updates, earnings reports, or any finance-related question."
            )
            base_tools.append(search)
            print("[INFO] Tavily web search enabled.")
        except Exception as e:
            print(f"[WARN] Tavily not available: {e}")
    else:
        print("[INFO] No TAVILY_API_KEY found — web search disabled.")
    return base_tools


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are FinBot, an expert AI finance assistant. You help users with:
- Stock prices and market data
- Cryptocurrency prices and trends
- Currency exchange rates
- Financial calculations (compound interest, ROI, etc.)
- General finance questions and news (if web search is available)

Always use the available tools to fetch real-time data before answering.
Be concise, accurate, and professional. Format numbers clearly.
If a user asks about multiple things, address each one systematically."""


def build_agent(openai_api_key: str):
    """Build and return the compiled LangGraph agent + memory."""
    os.environ["OPENAI_API_KEY"] = openai_api_key

    tools = _build_tools()
    llm   = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
    tool_node = ToolNode(tools)
    memory    = MemorySaver()

    # ── Nodes ──────────────────────────────────────────────────────────────
    def agent_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Conditional edge: route to tools or END."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── Graph ──────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # tool results loop back to agent

    compiled = graph.compile(checkpointer=memory)
    return compiled


# ══════════════════════════════════════════════════════════════════════════════
#  INVOKE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def chat(agent, user_message: str, thread_id: str = "default") -> str:
    """Send a message and return the assistant's reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content
