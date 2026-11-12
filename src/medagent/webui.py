"""答辩用的本地 WebUI + OpenEnv 风格会话 API

启动：
    python scripts/cli.py serve --port 8000

页面：http://127.0.0.1:8000/
API：
    GET  /api/cases
    POST /api/run    {"case_id": "...", "skill": 0.9}
    POST /api/reset  {"case_id": "...", "max_steps": 16}   -> session_id + observation
    POST /api/step   {"session_id": "...", "action": "<act>..."}

只依赖标准库，不引入前端构建链。
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .demo_agent import run_episode
from .env import CaseCatalog, MedAgentEnv
from .version import __version__

_CATALOG = CaseCatalog()
_SESSIONS: Dict[str, MedAgentEnv] = {}
_SESSION_LOCK = threading.Lock()
_MAX_SESSIONS = 200


def _remember(session_id: str, env: MedAgentEnv) -> None:
    with _SESSION_LOCK:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)))
        _SESSIONS[session_id] = env


def handle_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    case_id = payload.get("case_id")
    skill = float(payload.get("skill", 0.9))
    env = MedAgentEnv(env_config={"case_id": case_id})
    return asyncio.run(run_episode(env, skill=skill))


def handle_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    env = MedAgentEnv(env_config={
        "case_id": payload.get("case_id"),
        "max_steps": int(payload.get("max_steps", 16)),
    })
    observation, info, system_message = asyncio.run(env.reset())
    session_id = uuid.uuid4().hex[:12]
    _remember(session_id, env)
    return {
        "session_id": session_id,
        "observation": observation,
        "system_message": system_message,
        "info": info,
    }


def handle_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = payload.get("session_id", "")
    with _SESSION_LOCK:
        env = _SESSIONS.get(session_id)
    if env is None:
        raise KeyError(f"未知 session_id: {session_id}")

    observation, reward, done, info = asyncio.run(env.step(payload.get("action", "")))
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
        "summary": env.state.summary() if env.state else {},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"MedAgentUI/{__version__}"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._html(INDEX_HTML.replace("__VERSION__", __version__))
        elif path == "/api/cases":
            self._json({"cases": _CATALOG.list()})
        elif path == "/api/health":
            self._json({"ok": True, "version": __version__, "cases": len(_CATALOG.cases)})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        payload = self._read_json()
        if payload is None:
            return

        handlers = {
            "/api/run": handle_run,
            "/api/reset": handle_reset,
            "/api/step": handle_step,
        }
        handler = handlers.get(path)
        if handler is None:
            self._json({"error": "not found"}, status=404)
            return

        try:
            self._json(handler(payload))
        except KeyError as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # -- helpers ----------------------------------------------------------

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": f"invalid json: {exc}"}, status=400)
            return None
        if not isinstance(payload, dict):
            self._json({"error": "payload must be an object"}, status=400)
            return None
        return payload

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"MedAgent WebUI: http://{host}:{port}/  （Ctrl+C 停止）")
    print(f"会话 API: POST /api/reset  /api/step  |  演示 API: POST /api/run")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedAgent v__VERSION__</title>
<style>
  :root{
    --bg:#0f1419; --panel:#171d24; --panel2:#1e252e; --line:#2a333f;
    --text:#e6edf3; --muted:#8b98a5; --accent:#2dd4bf; --accent2:#38bdf8;
    --warn:#f59e0b; --bad:#f87171; --good:#34d399;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
  header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line);
         position:sticky;top:0;background:rgba(15,20,25,.92);backdrop-filter:blur(8px);z-index:5}
  header h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.3px}
  .badge{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px}
  .badge.on{color:var(--accent);border-color:rgba(45,212,191,.4)}
  .spacer{flex:1}
  main{display:grid;grid-template-columns:280px minmax(0,1fr) 320px;gap:16px;padding:16px 20px 40px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 10px}
  select,button,input{font:inherit}
  select{width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--line);
         border-radius:8px;padding:8px;margin-bottom:10px}
  .case-meta{font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:12px}
  .case-meta b{color:var(--text);font-weight:500}
  .row{display:flex;align-items:center;justify-content:space-between;font-size:12px;color:var(--muted);margin:10px 0 4px}
  input[type=range]{width:100%;accent-color:var(--accent)}
  button{width:100%;padding:10px;border-radius:8px;border:1px solid rgba(45,212,191,.4);
         background:rgba(45,212,191,.12);color:var(--accent);cursor:pointer;font-weight:600}
  button:hover{background:rgba(45,212,191,.2)}
  button:disabled{opacity:.5;cursor:progress}
  #transcript{height:calc(100vh - 190px);overflow:auto;padding-right:6px}
  .msg{margin-bottom:12px;display:flex;gap:10px}
  .msg .who{width:64px;flex:none;font-size:11px;color:var(--muted);padding-top:4px;text-align:right}
  .msg .bubble{flex:1;background:var(--panel2);border:1px solid var(--line);border-radius:10px;
               padding:10px 12px;white-space:pre-wrap;line-height:1.65;font-size:13px}
  .msg.assistant .bubble{border-left:3px solid var(--accent2)}
  .msg.user .bubble{border-left:3px solid var(--warn)}
  .msg.system .bubble{color:var(--muted);font-size:12px;max-height:120px;overflow:auto}
  .chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;margin-right:6px;
        background:rgba(56,189,248,.14);color:var(--accent2);border:1px solid rgba(56,189,248,.3)}
  .chip.good{background:rgba(52,211,153,.12);color:var(--good);border-color:rgba(52,211,153,.3)}
  .chip.bad{background:rgba(248,113,113,.12);color:var(--bad);border-color:rgba(248,113,113,.3)}
  .score-wrap{display:flex;align-items:center;gap:14px;margin-bottom:14px}
  .score-num{font-size:30px;font-weight:700;line-height:1}
  .score-num small{font-size:12px;color:var(--muted);font-weight:400}
  .bars{display:grid;gap:8px}
  .bar-row{font-size:12px;color:var(--muted)}
  .bar{height:8px;background:var(--panel2);border-radius:999px;overflow:hidden;margin-top:4px}
  .bar > i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));
           transition:width .6s ease}
  .kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
  .kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
  .kpi span{display:block;font-size:11px;color:var(--muted)}
  .kpi b{font-size:16px}
  .gt{margin-top:12px;font-size:12px;color:var(--muted);line-height:1.6;border-top:1px dashed var(--line);padding-top:10px}
  .gt b{color:var(--text)}
  details{margin-top:14px;font-size:12px;color:var(--muted)}
  details pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto}
  .empty{color:var(--muted);font-size:13px;text-align:center;padding:40px 0}
  @media (max-width:1100px){
    main{grid-template-columns:1fr}
    #transcript{height:420px}
  }
</style>
</head>
<body>
<header>
  <h1>MedAgent</h1>
  <span class="badge on">v__VERSION__</span>
  <span class="badge">mock 数据</span>
  <span class="badge">OpenEnv-style API</span>
  <div class="spacer"></div>
  <span class="badge" id="status">就绪</span>
</header>

<main>
  <section class="panel">
    <h2>病例</h2>
    <select id="case-select"></select>
    <div class="case-meta" id="case-meta">选择病例后点击运行</div>
    <div class="row"><span>医生水平</span><span id="skill-val">0.90</span></div>
    <input type="range" id="skill" min="0.2" max="1" step="0.05" value="0.9">
    <div class="row"><span>说明</span></div>
    <div class="case-meta">脚本化医生：问诊 → 检查 → 知识库 → 结论；水平越低，结论越含糊。</div>
    <button id="run">运行问诊</button>
  </section>

  <section class="panel">
    <h2>问诊轨迹</h2>
    <div id="transcript"><div class="empty">尚无轨迹</div></div>
  </section>

  <section class="panel">
    <h2>评分</h2>
    <div class="score-wrap">
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r="30" fill="none" stroke="#2a333f" stroke-width="7"/>
        <circle id="ring" cx="36" cy="36" r="30" fill="none" stroke="#2dd4bf" stroke-width="7"
                stroke-linecap="round" stroke-dasharray="188.5" stroke-dashoffset="188.5"
                transform="rotate(-90 36 36)" style="transition:stroke-dashoffset .6s ease"/>
      </svg>
      <div>
        <div class="score-num" id="score">0.00<small> / 1.00</small></div>
        <div class="case-meta" id="done-reason">未开始</div>
      </div>
    </div>
    <div class="bars" id="bars">
      <div class="bar-row">诊断<div class="bar"><i data-bar="diagnosis"></i></div></div>
      <div class="bar-row">治疗<div class="bar"><i data-bar="treatment"></i></div></div>
      <div class="bar-row">安全<div class="bar"><i data-bar="safety"></i></div></div>
      <div class="bar-row">效率<div class="bar"><i data-bar="efficiency"></i></div></div>
    </div>
    <div class="kpis">
      <div class="kpi"><span>轮次</span><b id="turns">-</b></div>
      <div class="kpi"><span>成本</span><b id="cost">-</b></div>
      <div class="kpi"><span>累计奖励</span><b id="reward">-</b></div>
      <div class="kpi"><span>工具调用</span><b id="calls">-</b></div>
    </div>
    <div class="gt" id="gt"></div>
    <details>
      <summary>会话 API（OpenEnv 风格）</summary>
      <pre>POST /api/reset {"case_id":"mock-mi-001","max_steps":16}
POST /api/step  {"session_id":"...","action":"&lt;act&gt;..."}
POST /api/run   {"case_id":"mock-mi-001","skill":0.9}</pre>
    </details>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s);

let cases = [];

async function loadCases(){
  const res = await fetch('/api/cases');
  const data = await res.json();
  cases = data.cases || [];
  const select = $('case-select');
  select.innerHTML = '';
  cases.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.case_id;
    option.textContent = `[${item.department}] ${item.chief_complaint.slice(0, 18)}`;
    select.appendChild(option);
  });
  updateMeta();
}

function updateMeta(){
  const item = cases.find((c) => c.case_id === $('case-select').value);
  if (!item) return;
  $('case-meta').innerHTML = `<b>${esc(item.case_id)}</b><br>${esc(item.chief_complaint)}<br>科室 ${esc(item.department)} · 难度 ${esc(item.difficulty)}`;
}

$('case-select').addEventListener('change', updateMeta);
$('skill').addEventListener('input', () => { $('skill-val').textContent = Number($('skill').value).toFixed(2); });

function actionOf(text){
  const match = /<action>\\s*([A-Za-z]+)\\s*<\\/action>/.exec(text || '');
  return match ? match[1].toUpperCase() : '';
}

function renderTranscript(transcript){
  const box = $('transcript');
  box.innerHTML = '';
  let lastAction = '';
  transcript.forEach((msg) => {
    if (msg.role === 'system' && !msg.reward) {
      const wrap = document.createElement('div');
      wrap.className = 'msg system';
      wrap.innerHTML = '<div class="who">system</div><div class="bubble"></div>';
      wrap.querySelector('.bubble').textContent = msg.content;
      box.appendChild(wrap);
      return;
    }
    if (msg.role === 'assistant') lastAction = actionOf(msg.content);
    const wrap = document.createElement('div');
    wrap.className = `msg ${msg.role}`;
    const who = msg.role === 'assistant' ? 'Agent' : (msg.role === 'user' ? 'Env' : 'sys');
    wrap.innerHTML = `<div class="who">${who}</div><div class="bubble"></div>`;
    const bubble = wrap.querySelector('.bubble');

    if (msg.role === 'assistant') {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = lastAction || 'ACTION';
      bubble.appendChild(chip);
      if (msg.reward != null) {
        const badge = document.createElement('span');
        badge.className = 'chip ' + (msg.reward >= 0 ? 'good' : 'bad');
        badge.textContent = (msg.reward >= 0 ? '+' : '') + Number(msg.reward).toFixed(3);
        bubble.appendChild(badge);
      }
      bubble.appendChild(document.createTextNode(msg.content));
    } else {
      bubble.textContent = msg.content;
    }
    box.appendChild(wrap);
  });
  box.scrollTop = box.scrollHeight;
}

function renderMetrics(data){
  const score = Number(data.score || 0);
  $('score').innerHTML = `${score.toFixed(2)}<small> / 1.00</small>`;
  $('done-reason').textContent = data.done_reason === 'final' ? '正常结束' : ('结束原因：' + data.done_reason);
  const circumference = 2 * Math.PI * 30;
  $('ring').style.strokeDashoffset = String(circumference * (1 - score));
  $('ring').style.stroke = score >= 0.75 ? '#34d399' : (score >= 0.4 ? '#f59e0b' : '#f87171');

  ['diagnosis','treatment','safety','efficiency'].forEach((key) => {
    const value = Number((data.breakdown || {})[key] || 0);
    const bar = document.querySelector(`[data-bar="${key}"]`);
    if (bar) bar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  });

  $('turns').textContent = data.turns;
  $('cost').textContent = `${Number(data.cost || 0).toFixed(0)} 元`;
  $('reward').textContent = Number(data.total_reward || 0).toFixed(2);
  const actions = (data.transcript || []).filter((m) => m.role === 'assistant')
    .map((m) => actionOf(m.content)).filter(Boolean);
  const callCount = actions.filter((a) => a !== 'FINAL').length;
  $('calls').textContent = `${callCount} 次`;

  const gt = data.ground_truth || {};
  $('gt').innerHTML = `<b>标准诊断</b>：${esc((gt.diagnosis||[]).join('、'))}<br>` +
                      `<b>标准处置</b>：${esc((gt.treatment||[]).join('；'))}<br>` +
                      `<b>禁忌</b>：${esc((gt.avoid||[]).join('、') || '无')}`;
}

async function run(){
  const button = $('run');
  button.disabled = true;
  $('status').textContent = '运行中';
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({case_id: $('case-select').value, skill: Number($('skill').value)}),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderTranscript(data.transcript || []);
    renderMetrics(data);
    $('status').textContent = '完成';
  } catch (err) {
    $('status').textContent = '失败';
    $('transcript').innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  } finally {
    button.disabled = false;
  }
}

$('run').addEventListener('click', run);
loadCases();
</script>
</body>
</html>
"""


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    serve(host, port)


if __name__ == "__main__":
    main()