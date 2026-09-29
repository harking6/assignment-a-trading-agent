from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from tools.market import MarketEnv


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8011


def load_env_file() -> None:
    for name in (".env", ".env.local"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
ENV = MarketEnv()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/state":
            self.send_json(ENV.to_dict())
            return
        if path == "/api/report":
            self.send_json({"report": ENV.report(), "trades": [t.public_dict() for t in ENV.trades]})
            return
        if path == "/api/quote":
            self.safe_json(lambda: {"quote": ENV.quote()})
            return
        if path == "/api/live/quote":
            self.safe_json(lambda: {"quote": ENV.live_quote()})
            return
        if path == "/api/paper/account":
            query = parse_qs(parsed.query)
            realtime = str(query.get("realtime", ["false"])[0]).lower() in {"1", "true", "yes"}
            self.safe_json(lambda: {"account": ENV.account(realtime), "report": ENV.report()})
            return
        self.serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self.read_json()
        if path == "/api/reset":
            self.safe_json(
                lambda: (
                    ENV.reset(
                        provider=str(payload.get("provider", ENV.config.provider)),
                        symbol=str(payload.get("symbol", ENV.config.symbol)),
                        benchmark_symbol=str(payload.get("benchmark_symbol", ENV.config.benchmark_symbol)),
                        price_adjust=str(payload.get("price_adjust", ENV.config.price_adjust)),
                        start=str(payload.get("start") or "") or None,
                        end=str(payload.get("end") or "") or None,
                        dataset=str(payload.get("dataset", ENV.dataset)),
                        strategy=str(payload.get("strategy", ENV.strategy)),
                        agent_mode=str(payload.get("agent_mode", ENV.config.agent_mode)),
                        llm_model=str(payload.get("llm_model", ENV.config.llm_model)),
                        deep_model=str(payload.get("deep_model", ENV.config.deep_model)),
                        llm_base_url=str(payload.get("llm_base_url") or "") or None,
                        max_debate_rounds=int(payload.get("max_debate_rounds", ENV.config.max_debate_rounds)),
                        max_risk_debate_rounds=int(payload.get("max_risk_debate_rounds", ENV.config.max_risk_debate_rounds)),
                        risk_budget=float(payload.get("risk_budget", ENV.risk_budget)),
                        fee_rate=float(payload.get("fee_rate", ENV.fee_rate)),
                        universe=payload.get("universe", ENV.config.universe),
                        dynamic_universe=bool(payload.get("dynamic_universe", ENV.config.dynamic_universe)),
                        dynamic_universe_limit=int(payload.get("dynamic_universe_limit", ENV.config.dynamic_universe_limit)),
                        max_positions=int(payload.get("max_positions", ENV.config.max_positions)),
                        top_n_sectors=int(payload.get("top_n_sectors", ENV.config.top_n_sectors)),
                        stocks_per_sector=int(payload.get("stocks_per_sector", ENV.config.stocks_per_sector)),
                        rebalance_frequency=int(payload.get("rebalance_frequency", ENV.config.rebalance_frequency)),
                        cash_reserve_ratio=float(payload.get("cash_reserve_ratio", ENV.config.cash_reserve_ratio)),
                        max_position_ratio=float(payload.get("max_position_ratio", ENV.config.max_position_ratio)),
                        initial_cash=float(payload.get("initial_cash", ENV.config.initial_cash)),
                        days=int(payload.get("days") or ENV.config.days or 0) or None,
                    )
                    or ENV.to_dict()
                )
            )
            return
        if path == "/api/step":
            self.safe_json(ENV.step)
            return
        if path == "/api/run":
            self.safe_json(ENV.run_to_end)
            return
        if path == "/api/paper/order":
            self.safe_json(
                lambda: ENV.manual_order(
                    side=str(payload.get("side", "hold")),
                    shares=int(payload.get("shares", 0)),
                    reason=str(payload.get("reason", "manual paper order")),
                    realtime=bool(payload.get("realtime", False)),
                )
            )
            return
        if path == "/api/paper/rebalance":
            self.safe_json(
                lambda: ENV.rebalance_to(
                    target_ratio=float(payload.get("target_ratio", 0.0)),
                    realtime=bool(payload.get("realtime", False)),
                    reason=str(payload.get("reason", "manual target exposure rebalance")),
                )
            )
            return
        if path == "/api/paper/adjust":
            self.safe_json(
                lambda: ENV.adjust_position(
                    direction=str(payload.get("direction", "increase")),
                    ratio_delta=float(payload.get("ratio_delta", 0.10)),
                    realtime=bool(payload.get("realtime", False)),
                )
            )
            return
        self.send_error(404, "Unknown API endpoint")

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def safe_json(self, fn) -> None:
        try:
            self.send_json(fn())
        except Exception as exc:
            self.send_json({"error": str(exc), "type": exc.__class__.__name__}, status=400)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, path: str) -> None:
        if path in ("", "/"):
            rel = "index.html"
        else:
            rel = unquote(path).lstrip("/")
        file_path = (ROOT / rel).resolve()
        if ROOT not in file_path.parents and file_path != ROOT:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[trading-agent] {self.address_string()} - {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Trading Agent server running at http://{HOST}:{PORT}/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
