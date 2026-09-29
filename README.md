# Assignment A: Trading Agent

This starter is a full paper-trading application with an intentionally missing
LangChain and LangGraph agent layer. Read [ASSIGNMENT.md](ASSIGNMENT.md) before
implementing the TODOs.

## Run with uv

```powershell
cd assignment-a-trading-agent
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8011/`.

Before the assignment is complete, the UI, quotes, account view, manual paper
orders, position adjustment, and benchmark metrics work. Agent step/run calls
return an `AssignmentTODO` JSON error that names the next missing component.

## Market Data

- `synthetic`: offline deterministic data, no API required.
- `csv`: local OHLCV files under `data/`.
- `akshare`: China A-share history and current quotes, no account required.
- `tushare`: China A-share history, requires `TUSHARE_TOKEN`.
- `yfinance`: US equities and ETFs, no account required.

For China symbols, `000001` means Ping An Bank while `000001.SH` or
`sh000001` means the SSE Composite Index. Historical A-share prices default to
forward-adjusted (`qfq`) data; current quotes are unadjusted market prices.

## Optional LLM Configuration

The complete assignment must run in `offline` mode without a key. To test an
OpenAI-compatible model, edit the local `.env` file:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_DEEP_MODEL=deepseek-reasoner
TRADING_AGENT_MODE=auto
```

`.env` is ignored and must not be published. The LLM proposes analysis and
trade intent only; risk controls and `SimulatedBroker` remain authoritative.

## Useful Commands

```powershell
python scripts/check_assignment.py
python -B -m unittest discover -s tests -v
```

Main files:

```text
agents/state.py             LangGraph typed state TODO
agents/graph.py             LangGraph workflow/checkpoint/routing TODOs
agents/llm_client.py        LangChain prompt/model/parser TODO
agents/memory.py            LangChain chat history TODO
tools/langchain_tools.py    LangChain tools TODO
tools/rag.py                LangChain RAG/retriever/runnable TODOs
engine/backtest.py          supplied event-driven engine
tools/broker.py             supplied simulated broker
app.py + index.html         supplied HTTP API and browser UI
```
