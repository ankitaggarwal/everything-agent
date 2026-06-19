"""The settings webpage -- a tiny stdlib HTTP server embedded in the agent.

No web framework, no new dependency: a ThreadingHTTPServer on a daemon thread,
serving a single-page UI at http://<robot>:8080 and a small JSON API backed by
the live EverythingAgent (status + recent transcript, swap backends/models, edit
persona, pick/preview voice, change trigger + toggle modules).

The agent passes itself in; the handler calls agent.web_status() / web_apply() /
web_say(). Anything that changes a backend/model writes the config file and asks
for a restart (POST /api/restart); trigger + persona apply live.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("everything_agent.web")

# Catalogue the UI offers (kept here so the page stays declarative).
AGENT_BACKENDS = ["fast", "claude_sdk", "haiku", "mock"]
TTS_MODELS = ["sonic-3", "sonic-2"]
ALL_MODULES = ["idle", "system_time", "emotions", "weather", "timers"]
VOICES = [
    {"id": "65209f8e-6140-4a20-b819-3cc2e21da19b", "name": "Nolan - Expressive (warm)"},
    {"id": "a0e99841-438c-4a64-b679-ae501e7d6091", "name": "Barbershop Man"},
    {"id": "79a125e8-cd45-4c13-8a67-188112f4dd22", "name": "British Lady"},
]


def start_web(agent, host: str = "0.0.0.0", port: int = 8080):
    handler = _make_handler(agent)
    httpd = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="settings-web")
    t.start()
    return httpd


def _make_handler(agent):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default per-request stderr spam
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj))

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:  # noqa: BLE001
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, _PAGE, "text/html; charset=utf-8")
            if self.path == "/api/status":
                try:
                    st = agent.web_status()
                    st["catalogue"] = {"agent_backends": AGENT_BACKENDS,
                                       "tts_models": TTS_MODELS,
                                       "all_modules": ALL_MODULES, "voices": VOICES}
                    return self._json(200, st)
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": str(e)})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            if self.path == "/api/apply":
                return self._json(200, agent.web_apply(body))
            if self.path == "/api/say":
                agent.web_say(body.get("text", ""))
                return self._json(200, {"ok": True})
            if self.path == "/api/restart":
                # Reply first, then restart (which kills this process; systemd
                # brings it straight back with the new config).
                self._json(200, {"ok": True, "restarting": True})
                threading.Thread(target=_restart, daemon=True).start()
                return
            return self._json(404, {"error": "not found"})

    return Handler


def _restart():
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", "everything-agent"],
                         start_new_session=True)
    except Exception:  # noqa: BLE001
        log.exception("restart failed")


_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reachy · Settings</title>
<style>
  :root{--bg:#0f1115;--card:#181b22;--line:#262b35;--fg:#e8ecf3;--mut:#8b94a7;
        --accent:#6ea8fe;--ok:#3ddc97;--warn:#ffb454}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;
    align-items:center;gap:12px;position:sticky;top:0;background:var(--bg);z-index:5}
  header h1{font-size:18px;margin:0;font-weight:650}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--mut)}
  .dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  main{max-width:860px;margin:0 auto;padding:22px}
  .tabs{display:flex;gap:6px;margin-bottom:18px;flex-wrap:wrap}
  .tabs button{background:var(--card);color:var(--mut);border:1px solid var(--line);
    padding:8px 14px;border-radius:9px;cursor:pointer;font-size:14px}
  .tabs button.active{color:var(--fg);border-color:var(--accent)}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:18px;margin-bottom:16px}
  .card h2{margin:0 0 14px;font-size:15px;font-weight:600}
  label{display:block;font-size:12px;color:var(--mut);margin:12px 0 5px;text-transform:uppercase;letter-spacing:.04em}
  input,select,textarea{width:100%;background:#0f1218;border:1px solid var(--line);
    color:var(--fg);border-radius:9px;padding:10px 12px;font:inherit}
  textarea{min-height:120px;resize:vertical}
  button.act{background:var(--accent);color:#06101f;border:none;padding:10px 16px;
    border-radius:9px;font-weight:600;cursor:pointer}
  button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin-top:8px}
  .row>div{flex:1;min-width:160px}
  .chips{display:flex;gap:8px;flex-wrap:wrap}
  .chip{padding:7px 12px;border-radius:20px;border:1px solid var(--line);cursor:pointer;color:var(--mut)}
  .chip.on{color:var(--fg);border-color:var(--ok);background:rgba(61,220,151,.08)}
  .turn{border-top:1px solid var(--line);padding:10px 0}
  .turn:first-child{border-top:none}
  .heard{color:var(--accent)} .reply{color:var(--fg)} .tmiss{color:var(--mut);font-size:12px}
  .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
    background:var(--ok);color:#06101f;padding:10px 18px;border-radius:10px;
    font-weight:600;opacity:0;transition:.3s;pointer-events:none}
  .toast.show{opacity:1}
  .hint{color:var(--mut);font-size:12px;margin-top:6px}
  .pane{display:none} .pane.active{display:block}
</style></head><body>
<header><span class="dot" id="dot"></span><h1>Reachy · Settings</h1>
  <span id="sub" class="hint" style="margin-left:auto"></span></header>
<main>
  <div class="tabs">
    <button data-t="status" class="active">Live status</button>
    <button data-t="brains">Backends & models</button>
    <button data-t="voice">Persona & voice</button>
    <button data-t="trigger">Trigger & modules</button>
  </div>

  <section class="pane active" id="status">
    <div class="card"><h2>Talk to it</h2>
      <div class="row"><div><input id="sayText" placeholder="Make the robot say this..."></div>
        <button class="act" onclick="say()">Speak</button></div>
      <div class="hint">Plays through the robot's speaker right now.</div></div>
    <div class="card"><h2>Recent conversation</h2><div id="turns">…</div></div>
  </section>

  <section class="pane" id="brains">
    <div class="card"><h2>Brains</h2>
      <label>Agent brain</label><select id="agent_backend"></select>
      <label>Agent model</label><input id="agent_model" placeholder="claude-haiku-4-5">
      <div class="hint">fast = snappy Haiku tool-calling · claude_sdk = MCP + Opus (slower).</div>
      <div style="margin-top:14px"><button class="act" onclick="apply(['agent_backend','agent_model'],true)">Save & restart</button></div>
    </div>
  </section>

  <section class="pane" id="voice">
    <div class="card"><h2>Personality</h2>
      <textarea id="persona"></textarea>
      <div class="hint">Applies live — no restart needed.</div>
      <div style="margin-top:12px"><button class="act" onclick="apply(['persona'],false)">Save persona</button></div>
    </div>
    <div class="card"><h2>Voice</h2>
      <label>Cartesia voice</label><select id="voice_id"></select>
      <label>TTS model</label><select id="tts_model"></select>
      <div class="row" style="margin-top:12px">
        <button class="act" onclick="apply(['voice_id','tts_model'],true)">Save & restart</button>
        <button class="ghost" onclick="say('Hi, this is how I sound.')">Preview current voice</button></div>
    </div>
  </section>

  <section class="pane" id="trigger">
    <div class="card"><h2>Wake name</h2>
      <label>Trigger word (say this to address it)</label><input id="trigger_word">
      <div class="hint">Applies live. Mis-hearings are matched automatically.</div>
      <div style="margin-top:12px"><button class="act" onclick="applyTrigger()">Save trigger</button></div>
    </div>
    <div class="card"><h2>Capabilities</h2><div class="chips" id="modchips"></div>
      <label>Default weather location</label><input id="weather_location" placeholder="London">
      <div style="margin-top:14px"><button class="act" onclick="applyModules()">Save & restart</button></div>
    </div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
let S={};
const $=id=>document.getElementById(id);
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); $(b.dataset.t).classList.add('active');});
function toast(m){const t=$('toast');t.textContent=m;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),1800);}
async function load(){
  try{S=await (await fetch('/api/status')).json();}catch(e){$('sub').textContent='offline';return;}
  $('dot').classList.toggle('on',S.online);
  $('sub').textContent=S.online?'online':'';
  const c=S.catalogue||{};
  fill('agent_backend',c.agent_backends,S.agent_backend);
  fill('tts_model',c.tts_models,S.tts_model);
  fillVoices(c.voices,S.voice_id);
  if(document.activeElement.id!=='agent_model')$('agent_model').value=S.agent_model||'';
  if(document.activeElement.id!=='persona')$('persona').value=S.persona||'';
  if(document.activeElement.id!=='trigger_word')$('trigger_word').value=S.trigger||'';
  if(document.activeElement.id!=='weather_location')$('weather_location').value=S.weather_location||'';
  chips(c.all_modules||[],S.modules||[]);
  turns(S.transcript||[]);
}
function fill(id,opts,sel){const e=$(id);if(!opts)return;e.innerHTML='';
  opts.forEach(o=>{const op=document.createElement('option');op.value=o;op.textContent=o;
    if(o===sel)op.selected=true;e.appendChild(op);});}
function fillVoices(vs,sel){const e=$('voice_id');if(!vs)return;e.innerHTML='';
  vs.forEach(v=>{const op=document.createElement('option');op.value=v.id;op.textContent=v.name;
    if(v.id===sel)op.selected=true;e.appendChild(op);});}
function chips(all,on){const e=$('modchips');e.innerHTML='';
  all.forEach(m=>{const d=document.createElement('div');d.className='chip'+(on.includes(m)?' on':'');
    d.textContent=m;d.onclick=()=>d.classList.toggle('on');e.appendChild(d);});}
function turns(ts){const e=$('turns');if(!ts.length){e.textContent='No conversation yet — say "hey reachy".';return;}
  e.innerHTML='';[...ts].reverse().forEach(t=>{const d=document.createElement('div');d.className='turn';
    d.innerHTML=`<div class="heard">🗣 ${esc(t.heard)}</div><div class="reply">🤖 ${esc(t.reply)}</div>`+
      `<div class="tmiss">stt ${t.stt}s · brain ${t.brain}s · tts ${t.tts}s · total ${t.total}s</div>`;
    e.appendChild(d);});}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function post(url,obj){return (await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(obj||{})})).json();}
async function say(t){const text=t||$('sayText').value;if(!text)return;await post('/api/say',{text});toast('Speaking…');}
async function apply(keys,restart){const ch={};keys.forEach(k=>ch[k]=$(k).value);
  const r=await post('/api/apply',ch);if(!r.ok){toast('Error: '+(r.error||'?'));return;}
  if(restart&&r.restart_needed){toast('Saved — restarting…');await post('/api/restart',{});}else toast('Saved ✓');}
async function applyTrigger(){const r=await post('/api/apply',{trigger:$('trigger_word').value});
  toast(r.ok?'Trigger updated ✓':'Error');}
async function applyModules(){const mods=[...document.querySelectorAll('#modchips .chip.on')].map(c=>c.textContent);
  const r=await post('/api/apply',{modules:mods,weather_location:$('weather_location').value});
  if(r.ok){toast('Saved — restarting…');await post('/api/restart',{});}else toast('Error');}
load();setInterval(load,4000);
</script></body></html>
"""
