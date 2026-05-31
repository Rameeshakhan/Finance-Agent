"""
Finance Assistant — Gradio UI
Run: python app.py
"""

import gradio as gr
import uuid
from agent import build_agent, chat

# ── Global agent instance (set after API key is provided) ──────────────────
_agent = None
_session_id = str(uuid.uuid4())   # unique session per app run

EXAMPLE_PROMPTS = [
    "What is the current price of Apple stock?",
    "How much is Bitcoin worth right now?",
    "What is the exchange rate from USD to PKR?",
    "If I invest $5000 at 8% annual return compounded yearly for 10 years, what will it be worth?",
    "Compare the price of Ethereum and Solana.",
    "What is 15% of 3,750?",
    "Show me TSLA and MSFT stock prices.",
    "Convert 1 EUR to JPY.",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def init_agent(api_key: str):
    global _agent, _session_id
    if not api_key.strip():
        return "⚠️ Please enter your OpenAI API key.", gr.update(interactive=False)
    try:
        _agent = build_agent(api_key.strip())
        _session_id = str(uuid.uuid4())
        return "✅ Agent ready! You can now chat.", gr.update(interactive=True)
    except Exception as e:
        return f"❌ Failed to initialize agent: {e}", gr.update(interactive=False)


def respond(user_message: str, history: list, thread_id: str):
    global _agent
    if not user_message.strip():
        return history, ""
    if _agent is None:
        history.append({"role": "assistant", "content": "⚠️ Please initialize the agent first by entering your OpenAI API key above."})
        return history, ""

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "⏳ Thinking..."})

    try:
        reply = chat(_agent, user_message, thread_id=thread_id)
    except Exception as e:
        reply = f"❌ Error: {e}"

    history[-1] = {"role": "assistant", "content": reply}
    return history, ""


def use_example(example: str):
    return example


def new_session():
    global _session_id
    _session_id = str(uuid.uuid4())
    return [], _session_id, "🆕 New session started. Memory cleared."


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
#chatbot { height: 520px; }
.status-box { font-size: 0.9em; padding: 6px 10px; border-radius: 6px; }
.example-btn { font-size: 0.82em !important; }
footer { display: none !important; }
"""

with gr.Blocks() as demo:

    # ── State ──────────────────────────────────────────────────────────────
    session_state = gr.State(_session_id)

    # ── Header ─────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 💹 FinBot — AI Finance Assistant
        **Powered by:** LangChain · LangGraph · GPT-4o  
        **Tools:** Stock Prices · Crypto Prices · Exchange Rates · Calculator · Web Search (optional)
        """
    )

    # ── API Key setup ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            api_key_input = gr.Textbox(
                label="🔑 OpenAI API Key",
                placeholder="sk-...",
                type="password",
                show_label=True,
            )
        with gr.Column(scale=1):
            init_btn = gr.Button("Initialize Agent", variant="primary")

    status_box = gr.Textbox(
        label="Status",
        value="Enter your API key and click 'Initialize Agent' to start.",
        interactive=False,
        elem_classes=["status-box"],
    )

    gr.Markdown("---")

    # ── Main layout ────────────────────────────────────────────────────────
    with gr.Row():
        # Chat column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                elem_id="chatbot",
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=finbot"),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about stocks, crypto, exchange rates, or calculations...",
                    label="Your message",
                    scale=5,
                    interactive=False,
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                new_session_btn = gr.Button("🆕 New Session", scale=1)

        # Sidebar column
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Prompts")
            for ex in EXAMPLE_PROMPTS:
                gr.Button(ex, elem_classes=["example-btn"]).click(
                    fn=use_example,
                    inputs=[gr.State(ex)],
                    outputs=[msg_input],
                )

            gr.Markdown("---")
            gr.Markdown(
                """
                ### 🛠️ Tools Available
                | Tool | Source |
                |------|--------|
                | 📈 Stock Price | Yahoo Finance |
                | 🪙 Crypto Price | CoinGecko |
                | 💱 Exchange Rate | Open ExchangeRate |
                | 🧮 Calculator | Built-in |
                | 🔍 Web Search | Tavily *(optional)* |

                ### 🧠 Memory
                Conversations persist within a session thread.  
                Click **New Session** to reset memory.

                ### 📌 Session ID
                """
            )
            session_display = gr.Textbox(
                value=_session_id[:8] + "...",
                label="Current Thread",
                interactive=False,
                max_lines=1,
            )

    # ── Event wiring ───────────────────────────────────────────────────────
    init_btn.click(
        fn=init_agent,
        inputs=[api_key_input],
        outputs=[status_box, msg_input],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, session_state],
        outputs=[chatbot, msg_input],
    )

    clear_btn.click(lambda: [], outputs=[chatbot])

    new_session_btn.click(
        fn=new_session,
        outputs=[chatbot, session_state, status_box],
    ).then(
        fn=lambda sid: sid[:8] + "...",
        inputs=[session_state],
        outputs=[session_display],
    )

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
