# 💹 FinBot — AI Finance Assistant

> An agentic AI application built with **LangChain**, **LangGraph**, and **Gradio**  
> that provides real-time financial information through intelligent tool use.

---

## 📌 Use Case

**Personal Finance Assistant** — answers questions about:
- Live stock prices (any ticker on Yahoo Finance)
- Cryptocurrency prices and market cap
- Currency exchange rates (150+ currencies)
- Financial calculations (compound interest, ROI, percentages)
- Financial news and research (optional, via Tavily web search)

---

## 🛠️ Tools Used

| Tool | Type | Source / API |
|------|------|-------------|
| `get_stock_price` | External API | Yahoo Finance (free, no key) |
| `get_crypto_price` | External API | CoinGecko (free, no key) |
| `get_exchange_rate` | External API | Open ExchangeRate API (free, no key) |
| `finance_calculator` | Custom Python | Built-in `math` module |
| `TavilySearchResults` | External API | Tavily Search *(optional)* |

---

## 🔗 APIs Integrated

| API | Key Required | Usage |
|-----|-------------|-------|
| Yahoo Finance (unofficial) | ❌ No | Stock quote data |
| CoinGecko API v3 | ❌ No | Crypto prices |
| Open ExchangeRate API | ❌ No | Forex rates |
| Tavily Search API | ✅ Optional | Web search for news |
| OpenAI API | ✅ Yes | GPT-4o as reasoning engine |

---

## 🧠 LangGraph Workflow

```
User Input
    │
    ▼
[agent node]  ←─────────────────────┐
    │                                │
    │  Has tool_calls?               │
    ├── YES → [tools node]  ─────────┘  (results loop back)
    │
    └── NO  → END (final response to user)
```

### State
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

### Nodes
- **`agent`** — GPT-4o with bound tools; decides whether to call a tool or respond
- **`tools`** — `ToolNode` executes whichever tool(s) the agent selected

### Conditional Routing
```python
def should_continue(state) -> str:
    if last_message.tool_calls:
        return "tools"   # route to ToolNode
    return END           # respond to user
```

The agent can chain multiple tool calls in one turn (e.g. fetch stock + calculate ROI).

---

## 🧠 Memory Implementation

Memory is implemented using **LangGraph's `MemorySaver`** checkpointer:

```python
memory = MemorySaver()
graph.compile(checkpointer=memory)
```

Each conversation is identified by a **`thread_id`** passed in the config:

```python
config = {"configurable": {"thread_id": "user-session-abc"}}
agent.invoke({"messages": [HumanMessage(...)]}, config=config)
```

### Graph State vs Memory
| | Graph State | Memory (Checkpointer) |
|--|------------|----------------------|
| Scope | Single invocation | Across multiple invocations |
| Stored | In-memory dict | Persisted checkpoint |
| Purpose | Pass data between nodes | Multi-turn conversation history |

Memory improves UX by allowing follow-up questions like  
*"What about its 5-year return?"* — the agent remembers you were discussing AAPL.

---

## 🚀 How to Run

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd finance_assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Either enter it directly in the Gradio UI, or set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### 4. (Optional) Enable web search

```bash
export TAVILY_API_KEY="tvly-..."
```

### 5. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:7860**

---

## 💬 Example Prompts

```
What is the current price of Apple stock?
How much is Bitcoin worth right now?
What is the exchange rate from USD to PKR?
If I invest $5000 at 8% annual return for 10 years, what will it be worth?
Compare Ethereum and Solana prices.
What is 15% of 3,750?
Show me TSLA and MSFT stock prices.
Convert 1 EUR to JPY.
```

---

## ⚠️ Challenges Faced

1. **Yahoo Finance API rate limits** — The unofficial endpoint can occasionally throttle requests; added proper headers and error handling to handle this gracefully.
2. **Tool chaining** — Ensuring the LangGraph loop correctly returned tool results back to the agent required understanding the `add_messages` reducer and the `tools → agent` edge.
3. **Safe calculator** — Implementing a safe `eval()` that blocks access to builtins while allowing `math` functions.
4. **Gradio state management** — Managing per-session `thread_id` so each user gets isolated memory.

---

## 🔮 Future Improvements

- [ ] Portfolio tracker (buy/sell history, P&L)
- [ ] LangSmith tracing for debugging tool calls
- [ ] Streaming responses in Gradio
- [ ] Persistent SQLite checkpointer (replace in-memory)
- [ ] Multi-agent: separate "news agent" and "data agent" with a supervisor
- [ ] Human-in-the-loop confirmation before executing large calculations
- [ ] Deployment on Hugging Face Spaces
- [ ] RAG over financial reports (PDF ingestion)

---

## 📊 Evaluation Coverage

| Component | Implementation |
|-----------|---------------|
| Use case design | Finance assistant with 5 tools |
| LangGraph workflow | StateGraph with agent + ToolNode, conditional routing |
| Tool integration | 4 tools (5 with Tavily) |
| API integration | Yahoo Finance, CoinGecko, Open ExchangeRate, Tavily |
| Conditional routing | `should_continue()` routes to tools or END |
| Memory | `MemorySaver` with thread-based checkpointing |
| Gradio interface | Full chat UI with examples and session management |
| Documentation | This README |
