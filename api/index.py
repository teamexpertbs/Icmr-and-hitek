import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HF_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/llmsunny/icrm-hitek-full-db-mixedcraka/resolve/main",
).rstrip("/")

SEARCH_FIELDS = [
    "name","fathersName","phoneNumber","aadharNumber","otherNumber",
    "address","district","pincode","state","town","source",
]
NUMBER_FIELDS = ["phoneNumber","aadharNumber","otherNumber"]
DUPLICATE_CAP = 2

PHONE_URLS  = [f"{HF_BASE}/idx_phone.{i}.parquet"  for i in range(7)]
AADHAR_URLS = [f"{HF_BASE}/idx_aadhar.{i}.parquet" for i in range(7)]

pool  = ThreadPoolExecutor(max_workers=4)
_lock = threading.Lock()
_tloc = threading.local()
_conns: list = []


def _make_conn():
    import duckdb
    c = duckdb.connect()
    c.execute("SET home_directory='/tmp'")
    c.execute("SET extension_directory='/tmp/dkx'")
    c.execute("SET enable_progress_bar=false")
    c.execute("SET threads=2")
    # No memory limit — use full container RAM
    for ext in ("parquet", "httpfs"):
        try:
            c.execute(f"LOAD {ext}")
        except Exception:
            try:
                c.execute(f"INSTALL {ext}; LOAD {ext}")
            except Exception:
                pass
    c.execute("SET http_timeout=30000")
    c.execute("SET http_keep_alive=true")
    # Sorted index views — DuckDB does binary search, reads only matching row groups
    pl = ", ".join(f"'{u}'" for u in PHONE_URLS)
    al = ", ".join(f"'{u}'" for u in AADHAR_URLS)
    c.execute(f"CREATE OR REPLACE VIEW v_phone  AS SELECT * FROM read_parquet([{pl}])")
    c.execute(f"CREATE OR REPLACE VIEW v_aadhar AS SELECT * FROM read_parquet([{al}])")
    return c


def _conn():
    tid = getattr(_tloc, "id", None)
    if tid is None:
        with _lock:
            tid = len(_conns)
            _tloc.id = tid
    with _lock:
        while len(_conns) <= tid:
            _conns.append(_make_conn())
    return _conns[tid]


def _person_key(r):
    ph = (r.get("phoneNumber") or "").strip()
    ad = (r.get("aadharNumber") or "").strip()
    return (ph, ad) if ph or ad else (
        (r.get("name") or "").strip(),
        (r.get("fathersName") or "").strip()
    )


def _nums(r):
    seen, out = set(), []
    for f in NUMBER_FIELDS:
        v = str(r.get(f) or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append({"field": f, "value": v})
    return out


def _dedup(rows, lim):
    s: dict = {}
    res = []
    for r in rows:
        k = _person_key(r)
        if s.get(k, 0) < DUPLICATE_CAP:
            s[k] = s.get(k, 0) + 1
            rec = dict(r)
            rec["connected_numbers"] = _nums(rec)
            res.append(rec)
        if len(res) >= lim:
            break
    return res


def _search(field: str, value: str, limit: int) -> dict:
    if field == "phoneNumber":
        view = "v_phone"
    elif field == "aadharNumber":
        view = "v_aadhar"
    else:
        return {"field": field, "value": value, "count": 0, "results": []}
    v = value.replace("'", "''")
    sql = f"SELECT * FROM {view} WHERE {field}='{v}' LIMIT {limit * DUPLICATE_CAP + 10}"
    c = _conn()
    rows = c.execute(sql).fetchall()
    cols = [d[0] for d in c.description]
    res = _dedup([dict(zip(cols, r)) for r in rows], limit)
    return {"field": field, "value": value, "count": len(res), "results": res}


def _auto(q: str, limit: int) -> dict:
    q = q.strip()
    if not (q.isdigit() and len(q) >= 8):
        return {"query": q, "searched_fields": [], "count": 0, "results": []}
    r = _search("phoneNumber", q, limit)
    searched = ["phoneNumber"]
    if not r["results"]:
        r = _search("aadharNumber", q, limit)
        searched.append("aadharNumber")
    return {"query": q, "searched_fields": searched, "count": r["count"], "results": r["results"]}


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="ICMR + HITEK Search API", version="2.0")


@app.get("/", response_class=HTMLResponse)
def idx():
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICMR + HITEK Search</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0d0d1a;color:#e0e0f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#16162a;border:1px solid #2a2a50;border-radius:20px;padding:44px;max-width:720px;width:100%;box-shadow:0 12px 48px rgba(0,0,0,.5)}
h1{font-size:2.2rem;background:linear-gradient(135deg,#7c6fff,#ff6b9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px}
.sub{color:#7070a0;margin-bottom:20px;font-size:.95rem}
.row{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
input{flex:1;min-width:200px;padding:13px 18px;background:#0d0d1a;border:1.5px solid #33335a;border-radius:10px;color:#e0e0f0;font-size:1rem;outline:none;transition:.2s}
input:focus{border-color:#7c6fff}
button{padding:13px 26px;background:linear-gradient(135deg,#7c6fff,#ff6b9d);border:none;border-radius:10px;color:#fff;font-size:1rem;font-weight:700;cursor:pointer}
select{padding:13px 14px;background:#0d0d1a;border:1.5px solid #33335a;border-radius:10px;color:#e0e0f0;font-size:.9rem;outline:none}
#res{background:#0d0d1a;border:1px solid #2a2a50;border-radius:12px;padding:18px;font-family:monospace;font-size:.82rem;white-space:pre-wrap;word-break:break-all;display:none;max-height:500px;overflow-y:auto;margin-top:14px;color:#88ff88}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.chip{background:#22223a;padding:5px 14px;border-radius:20px;font-size:.78rem;color:#8888c0}
.foot{margin-top:22px;text-align:center;color:#444;font-size:.78rem}
.foot a{color:#7c6fff;text-decoration:none}
.info{background:#0a1a0a;border:1px solid #1a6a1a;border-radius:8px;padding:10px 14px;font-size:.8rem;color:#66ff88;margin-bottom:16px}
</style></head>
<body><div class="card">
<h1>🔍 ICMR + HITEK API</h1>
<p class="sub">5 Billion Records — Phone & Aadhaar Lookup</p>
<div class="info">✅ Full server — No timeout! Phone ~3s | Aadhaar ~10s</div>
<div class="chips">
  <span class="chip">📱 Phone</span>
  <span class="chip">🪪 Aadhaar</span>
  <span class="chip">🆓 Free</span>
</div>
<div class="row">
  <input id="q" type="text" placeholder="Phone ya Aadhaar number daalo..." onkeydown="if(event.key==='Enter')go()"/>
  <select id="f">
    <option value="">Auto</option>
    <option value="phoneNumber">📱 Phone</option>
    <option value="aadharNumber">🪪 Aadhaar</option>
  </select>
  <button onclick="go()">Search</button>
</div>
<div id="res"></div>
<div class="foot">👨‍💻 @kzr0x · <a href="/docs">API Docs</a> · <a href="/health">Status</a></div>
</div>
<script>
async function go(){
  const q=document.getElementById('q').value.trim();
  const f=document.getElementById('f').value;
  const r=document.getElementById('res');
  if(!q)return;
  r.style.display='block';r.style.color='#7c6fff';
  r.textContent='⏳ Searching...';
  const url=f?`/search?field=${f}&q=${encodeURIComponent(q)}`:`/search?q=${encodeURIComponent(q)}`;
  try{
    const d=await fetch(url);
    const t=await d.text();
    r.style.color=d.ok?'#88ff88':'#ff8888';
    r.textContent=JSON.stringify(JSON.parse(t),null,2);
  }catch(e){r.style.color='#ff8888';r.textContent='❌ '+e.message;}
}
</script></body></html>"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": "llmsunny/icrm-hitek-full-db-mixedcraka",
        "records": 5_009_587_740,
        "phone_search": "~3-5s",
        "aadhaar_search": "~10-20s",
    }


@app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
    pretty: bool = Query(True),
):
    qv = (q or mobile or "").strip()
    if not qv:
        raise HTTPException(422, "Provide q= param")
    loop = asyncio.get_running_loop()
    try:
        fn = (_search if field else _auto)
        args = (field, qv, limit) if field else (qv, limit)
        data = await loop.run_in_executor(pool, fn, *args)
        if field:
            data.update({"query": qv, "searched_fields": [field]})
    except Exception as e:
        raise HTTPException(500, str(e))
    out = {
        "success": bool(data["count"]),
        **data,
        "number": qv,
        "total": data["count"],
    }
    return Response(
        json.dumps(out, indent=2 if pretty else None, ensure_ascii=False),
        media_type="application/json",
    )


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "empty queries")
    if len(req.queries) > 20:
        raise HTTPException(400, "max 20 queries")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    safe = [
        r if not isinstance(r, Exception) else {"error": str(r)}
        for r in results
    ]
    return Response(
        json.dumps({"searches": len(req.queries), "results": safe}, indent=2),
        media_type="application/json",
    )
