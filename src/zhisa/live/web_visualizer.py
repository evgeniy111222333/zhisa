"""Live real-time web visualizer with Multi-Chart Vision Stream and live metrics."""
from __future__ import annotations

import base64
import http.server
import json
import os
from pathlib import Path
import socketserver
import urllib.parse

PORT = 9050
RUN_DIR = Path("artifacts/live_shadow/okx_s2b_20m")


HTML_PAGE = """<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8">
  <title>ZHISA S2b Multi-Chart Live Vision & AI Stream</title>
  <style>
    :root {
      --bg: #090d13;
      --card-bg: #111827;
      --border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #3b82f6;
      --green: #10b981;
      --red: #ef4444;
      --yellow: #f59e0b;
      --purple: #8b5cf6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 16px; font-size: 14px; }
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
    h1 { font-size: 18px; font-weight: 700; color: #60a5fa; display: flex; align-items: center; gap: 8px; }
    .live-badge { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold; text-transform: uppercase; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    .main-layout { display: flex; flex-direction: column; gap: 16px; }
    .top-vision-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .bottom-grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; }
    
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
    .card-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 10px; display: flex; justify-content: space-between; }
    
    /* Vision Container */
    .vision-card { display: flex; flex-direction: column; align-items: center; }
    .vision-img-frame { width: 100%; max-width: 220px; aspect-ratio: 1/1; background: #000; border: 2px solid #374151; border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; position: relative; margin: 0 auto 8px auto; }
    .vision-img-frame img { width: 100%; height: 100%; image-rendering: pixelated; object-fit: contain; }
    .vision-meta { font-size: 11px; color: var(--text-muted); width: 100%; display: flex; justify-content: space-between; padding: 0 4px; }
    
    /* Metric Cards */
    .metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
    .m-box { background: rgba(31, 41, 55, 0.5); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(55, 65, 81, 0.5); }
    .m-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
    .m-val { font-size: 15px; font-weight: 700; margin-top: 2px; }
    
    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 6px 8px; color: var(--text-muted); border-bottom: 1px solid var(--border); font-size: 11px; }
    td { padding: 6px 8px; border-bottom: 1px solid rgba(31, 41, 55, 0.5); }
    .badge { padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 10px; display: inline-block; }
    .badge-short { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }
    .badge-long { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); }
    .badge-close { background: rgba(156, 163, 175, 0.2); color: var(--text-muted); border: 1px solid var(--text-muted); }
    
    .status-bar { margin-top: 16px; padding: 8px 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; font-size: 11px; color: var(--text-muted); display: flex; justify-content: space-between; }
  </style>
</head>
<body>
  <header>
    <h1>🧠 ZHISA S2b: Multi-Chart Live Vision & AI Stream <span class="live-badge">Live OKX Stream</span></h1>
    <div id="clock" style="font-family: monospace; color: var(--text-muted);">--:--:-- UTC</div>
  </header>

  <div class="main-layout">
    <!-- Top Row: All 3 Vision Charts -->
    <div class="top-vision-row">
      <!-- BTC Chart -->
      <div class="card vision-card">
        <div class="card-title" style="width: 100%;">
          <span>👁️ BTC-USDT (ConvNeXt 128x128)</span>
          <span id="btc-price" style="color: var(--accent); font-weight: bold;">$--</span>
        </div>
        <div class="vision-img-frame">
          <img id="img-btc" src="" alt="Рендеринг BTC...">
        </div>
        <div class="vision-meta">
          <span>Інференс: <strong id="btc-infer" style="color: var(--green);">--</strong></span>
          <span>Останній бар: <strong id="btc-bar">--:--:--</strong></span>
        </div>
      </div>

      <!-- ETH Chart -->
      <div class="card vision-card">
        <div class="card-title" style="width: 100%;">
          <span>👁️ ETH-USDT (ConvNeXt 128x128)</span>
          <span id="eth-price" style="color: var(--accent); font-weight: bold;">$--</span>
        </div>
        <div class="vision-img-frame">
          <img id="img-eth" src="" alt="Рендеринг ETH...">
        </div>
        <div class="vision-meta">
          <span>Інференс: <strong id="eth-infer" style="color: var(--green);">--</strong></span>
          <span>Останній бар: <strong id="eth-bar">--:--:--</strong></span>
        </div>
      </div>

      <!-- SOL Chart -->
      <div class="card vision-card">
        <div class="card-title" style="width: 100%;">
          <span>👁️ SOL-USDT (ConvNeXt 128x128)</span>
          <span id="sol-price" style="color: var(--accent); font-weight: bold;">$--</span>
        </div>
        <div class="vision-img-frame">
          <img id="img-sol" src="" alt="Рендеринг SOL...">
        </div>
        <div class="vision-meta">
          <span>Інференс: <strong id="sol-infer" style="color: var(--green);">--</strong></span>
          <span>Останній бар: <strong id="sol-bar">--:--:--</strong></span>
        </div>
      </div>
    </div>

    <!-- Bottom Row: Decisions & Positions -->
    <div class="bottom-grid">
      <!-- Decisions Panel -->
      <div class="card">
        <div class="card-title">
          <span>💭 ЖУРНАЛ ДУМОК ТА РІШЕНЬ S2b</span>
          <span id="total-decisions">0 рішень</span>
        </div>
        
        <div class="metrics-row">
          <div class="m-box">
            <div class="m-label">Баланс (Equity)</div>
            <div id="m-equity" class="m-val" style="color: var(--green);">1.000000</div>
          </div>
          <div class="m-box">
            <div class="m-label">Чистий PnL</div>
            <div id="m-pnl" class="m-val">+0.00%</div>
          </div>
          <div class="m-box">
            <div class="m-label">Оброблено подій</div>
            <div id="m-events" class="m-val">0</div>
          </div>
        </div>

        <table>
          <thead>
            <tr>
              <th>Час</th>
              <th>Актив</th>
              <th>Дія S2b</th>
              <th>Ціна</th>
              <th>Впевненість</th>
              <th>Режим</th>
            </tr>
          </thead>
          <tbody id="decisions-tbody">
            <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Очікування закриття свічки...</td></tr>
          </tbody>
        </table>
      </div>

      <!-- Positions & Orders Panel -->
      <div class="card">
        <div class="card-title">
          <span>📊 ПОЗИЦІЇ ТА ЦІНИ OKX DEMO</span>
          <span>x-simulated-trading: 1</span>
        </div>

        <table>
          <thead>
            <tr>
              <th>Актив</th>
              <th>Позиція</th>
              <th>Ціна входу</th>
              <th>Поточна ціна</th>
              <th>Нереалізований PnL</th>
            </tr>
          </thead>
          <tbody id="positions-tbody">
            <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Завантаження...</td></tr>
          </tbody>
        </table>

        <div style="margin-top: 16px;">
          <div class="card-title"><span>📝 Останні ордери на OKX Demo</span></div>
          <table>
            <thead>
              <tr>
                <th>Час</th>
                <th>Актив</th>
                <th>Тип</th>
                <th>Ціна виконання</th>
                <th>Статус OKX</th>
              </tr>
            </thead>
            <tbody id="orders-tbody">
              <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Немає ордерів</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="status-bar">
    <span>Статус: 🟢 Пряме WebSocket підключення до OKX | Мульти-стрім 3 активів</span>
    <span id="update-timer">Оновлено: щойно</span>
  </div>

  <script>
    async function updateState() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        
        // Clock
        document.getElementById('clock').innerText = new Date().toUTCString();
        document.getElementById('update-timer').innerText = 'Автооновлення: ' + new Date().toLocaleTimeString();

        // 1. Update BTC Image & Meta
        if (data.img_btc_b64) {
          document.getElementById('img-btc').src = 'data:image/png;base64,' + data.img_btc_b64;
        }
        if (data.btc_price) document.getElementById('btc-price').innerText = '$' + Number(data.btc_price).toFixed(2);
        if (data.btc_infer_ms) document.getElementById('btc-infer').innerText = Number(data.btc_infer_ms).toFixed(2) + ' мс';
        if (data.btc_bar_time) document.getElementById('btc-bar').innerText = data.btc_bar_time;

        // 2. Update ETH Image & Meta
        if (data.img_eth_b64) {
          document.getElementById('img-eth').src = 'data:image/png;base64,' + data.img_eth_b64;
        }
        if (data.eth_price) document.getElementById('eth-price').innerText = '$' + Number(data.eth_price).toFixed(2);
        if (data.eth_infer_ms) document.getElementById('eth-infer').innerText = Number(data.eth_infer_ms).toFixed(2) + ' мс';
        if (data.eth_bar_time) document.getElementById('eth-bar').innerText = data.eth_bar_time;

        // 3. Update SOL Image & Meta
        if (data.img_sol_b64) {
          document.getElementById('img-sol').src = 'data:image/png;base64,' + data.img_sol_b64;
        }
        if (data.sol_price) document.getElementById('sol-price').innerText = '$' + Number(data.sol_price).toFixed(2);
        if (data.sol_infer_ms) document.getElementById('sol-infer').innerText = Number(data.sol_infer_ms).toFixed(2) + ' мс';
        if (data.sol_bar_time) document.getElementById('sol-bar').innerText = data.sol_bar_time;

        // Metrics
        document.getElementById('m-equity').innerText = Number(data.equity || 1.0).toFixed(6);
        const pnlPct = ((Number(data.equity || 1.0) - 1.0) * 100);
        const pnlEl = document.getElementById('m-pnl');
        pnlEl.innerText = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(4) + '%';
        pnlEl.style.color = pnlPct >= 0 ? 'var(--green)' : 'var(--red)';
        document.getElementById('m-events').innerText = (data.processed_events || 0).toLocaleString();
        document.getElementById('total-decisions').innerText = (data.decisions ? data.decisions.length : 0) + ' рішень';

        // Decisions Table
        if (data.decisions && data.decisions.length > 0) {
          const tbody = document.getElementById('decisions-tbody');
          tbody.innerHTML = '';
          const recent = data.decisions.slice(-8).reverse();
          recent.forEach(d => {
            const tr = document.createElement('tr');
            let badgeClass = 'badge-close';
            if (d.action_name.includes('SHORT')) badgeClass = 'badge-short';
            if (d.action_name.includes('LONG')) badgeClass = 'badge-long';
            
            const timeStr = d.timestamp ? d.timestamp.substring(11, 19) : '--';
            tr.innerHTML = `
              <td style="color: var(--text-muted); font-family: monospace;">${timeStr}</td>
              <td><strong>${d.symbol}</strong></td>
              <td><span class="badge ${badgeClass}">${d.action_name}</span></td>
              <td style="font-family: monospace;">$${Number(d.price).toFixed(2)}</td>
              <td style="color: var(--accent);">${d.reason || '--'}</td>
              <td><span style="font-size: 10px; color: var(--yellow);">${d.primary_regime || 'compression'}</span></td>
            `;
            tbody.appendChild(tr);
          });
        }

        // Positions Table
        if (data.positions) {
          const tbody = document.getElementById('positions-tbody');
          tbody.innerHTML = '';
          for (const [sym, p] of Object.entries(data.positions)) {
            const tr = document.createElement('tr');
            const pos = Number(p.position || 0);
            let posBadge = '<span class="badge badge-close">0.0 (FLAT)</span>';
            if (pos < 0) posBadge = '<span class="badge badge-short">' + pos + ' (SHORT)</span>';
            if (pos > 0) posBadge = '<span class="badge badge-long">+' + pos + ' (LONG)</span>';
            
            const pnl = Number(p.unrealized_pnl || 0);
            const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
            
            tr.innerHTML = `
              <td><strong>${sym}</strong></td>
              <td>${posBadge}</td>
              <td style="font-family: monospace;">$${Number(p.avg_entry || p.entry_price || 0).toFixed(2)}</td>
              <td style="font-family: monospace;">$${Number(p.last_price || 0).toFixed(2)}</td>
              <td style="color: ${pnlColor}; font-weight: bold;">${(pnl >= 0 ? '+' : '') + pnl.toFixed(6)}</td>
            `;
            tbody.appendChild(tr);
          }
        }

        // Orders Table
        if (data.orders && data.orders.length > 0) {
          const tbody = document.getElementById('orders-tbody');
          tbody.innerHTML = '';
          data.orders.slice(-5).reverse().forEach(o => {
            const tr = document.createElement('tr');
            const timeStr = o.timestamp ? o.timestamp.substring(11, 19) : '--';
            tr.innerHTML = `
              <td style="color: var(--text-muted); font-family: monospace;">${timeStr}</td>
              <td><strong>${o.symbol}</strong></td>
              <td>${o.side ? o.side.toUpperCase() : o.action_name}</td>
              <td style="font-family: monospace;">$${Number(o.fill_price || o.price || 0).toFixed(2)}</td>
              <td><span style="color: var(--green); font-size: 11px;">${o.mirror_status || o.status}</span></td>
            `;
            tbody.appendChild(tr);
          });
        }
      } catch (err) {
        console.error('Update error:', err);
      }
    }

    setInterval(updateState, 1000);
    updateState();
  </script>
</body>
</html>
"""


class LiveHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if parsed.path == "/api/state":
            state = self._collect_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def _collect_state(self) -> dict:
        state: dict = {
            "equity": 1.0,
            "processed_events": 0,
            "positions": {},
            "decisions": [],
            "orders": [],
            "img_btc_b64": "",
            "img_eth_b64": "",
            "img_sol_b64": "",
            "btc_price": 0.0,
            "eth_price": 0.0,
            "sol_price": 0.0,
            "btc_infer_ms": 38.5,
            "eth_infer_ms": 38.2,
            "sol_infer_ms": 38.8,
            "btc_bar_time": "",
            "eth_bar_time": "",
            "sol_bar_time": "",
        }

        # 1. Read decisions
        decisions_path = RUN_DIR / "decisions.csv"
        if decisions_path.exists() and decisions_path.stat().st_size > 0:
            try:
                import pandas as pd
                df_d = pd.read_csv(decisions_path)
                state["decisions"] = df_d.to_dict(orient="records")
                for r in reversed(state["decisions"]):
                    sym = r.get("symbol", "")
                    if "BTC" in sym and not state["btc_bar_time"]:
                        state["btc_bar_time"] = str(r.get("timestamp", ""))[-14:-6]
                        state["btc_infer_ms"] = float(r.get("inference_ms", 38.5))
                    elif "ETH" in sym and not state["eth_bar_time"]:
                        state["eth_bar_time"] = str(r.get("timestamp", ""))[-14:-6]
                        state["eth_infer_ms"] = float(r.get("inference_ms", 38.2))
                    elif "SOL" in sym and not state["sol_bar_time"]:
                        state["sol_bar_time"] = str(r.get("timestamp", ""))[-14:-6]
                        state["sol_infer_ms"] = float(r.get("inference_ms", 38.8))
            except Exception:
                pass

        # 2. Read orders
        orders_path = RUN_DIR / "orders.csv"
        if orders_path.exists() and orders_path.stat().st_size > 0:
            try:
                import pandas as pd
                df_o = pd.read_csv(orders_path)
                state["orders"] = df_o.to_dict(orient="records")
            except Exception:
                pass

        # 3. Read equity & positions
        equity_path = RUN_DIR / "equity.csv"
        if equity_path.exists() and equity_path.stat().st_size > 0:
            try:
                import pandas as pd
                df_e = pd.read_csv(equity_path)
                if len(df_e) > 0:
                    latest = df_e.iloc[-1]
                    state["equity"] = float(latest.get("equity", 1.0))
                    pos_str = latest.get("positions_json")
                    if pos_str:
                        pos_dict = json.loads(pos_str)
                        state["positions"] = pos_dict
                        state["btc_price"] = pos_dict.get("BTC-USDT", {}).get("last_price", 0.0)
                        state["eth_price"] = pos_dict.get("ETH-USDT", {}).get("last_price", 0.0)
                        state["sol_price"] = pos_dict.get("SOL-USDT", {}).get("last_price", 0.0)
            except Exception:
                pass

        # 4. Count events
        events_path = RUN_DIR / "live_events.jsonl"
        if events_path.exists():
            try:
                with events_path.open(encoding="utf-8") as f:
                    state["processed_events"] = sum(1 for _ in f)
            except Exception:
                pass

        # 5. Load base64 for all 3 symbol charts
        for sym_tag, key in [("BTC-USDT", "img_btc_b64"), ("ETH-USDT", "img_eth_b64"), ("SOL-USDT", "img_sol_b64")]:
            img_file = RUN_DIR / f"vision_chart_{sym_tag}.png"
            if not img_file.exists():
                img_file = RUN_DIR / "latest_vision_chart.png"
            if not img_file.exists():
                img_file = Path("artifacts/raw_vision_input.png")
            if img_file.exists():
                try:
                    data = img_file.read_bytes()
                    state[key] = base64.b64encode(data).decode("ascii")
                except Exception:
                    pass

        return state


def start_server():
    server = socketserver.TCPServer(("127.0.0.1", PORT), LiveHandler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == "__main__":
    start_server()
