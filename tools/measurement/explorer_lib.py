"""explorer_lib: the shared builder for Marino's HTML explorers.

Why this exists (11 Oct 2026). Fourteen explorer folders and twelve `build_explorer*.py`
generators were found across the repos, each re-deciding the same things: the page frame,
the CSS, sortable tables, where the data goes, how figures are laid out, how a sequence of
checkpoints or sessions is scrubbed through, how the page links back to the wiki. This module
holds those decisions once. A new explorer imports it and writes only what is specific to its
dataset: which pages exist and what goes on them.

Design rules it enforces (see SKILL.md for the reasoning):
  - Static HTML. Opens from disk with file:// and from the deck server alike, so nothing may
    use fetch() or XMLHttpRequest: data that a page needs at runtime is embedded as a <script>
    payload (see plotly_page) and figures are PNG files next to the page.
  - No external assets except, optionally, the Plotly CDN; pass plotly="local" to copy the
    local Plotly from rat_strategy_discovery/explorer/lib instead, for pages that must work
    offline.
  - One CSS, light by default (matplotlib figures have white backgrounds), dark on request.
  - Every page carries the site nav, a build timestamp and a back-link to its wiki page.
  - Tables are sortable (click a header) and filterable (type in the box) with a few lines of
    inline JS and nothing else.
  - A `slider` scrubs through an image sequence (checkpoints, sessions, conditions) with a
    range input, arrow keys and a play button; images are preloaded.

Usage sketch:

    from explorer_lib import Site, table, fig, slider, sparkline, kv, note, caveat
    site = Site("out/", "My explorer", nav=[("index.html", "Overview"), ("rats.html", "Rats")],
                wiki_html="file:///Users/mpagan/Documents/GitHub/_master/html/wiki/projects/x.html")
    site.page("index.html", "Overview", body_html, lead="one sentence")
    site.finish()        # writes nothing more; kept for symmetry and the build log

Everything returns HTML strings, so pages are composed by concatenation. html.escape() is the
caller's job for untrusted strings; the helpers escape what they are told to.
"""
from __future__ import annotations

import html as _html
import json
import os
import shutil
import time

# --------------------------------------------------------------------------------------- CSS

_CSS_LIGHT = """
:root{--bg:#fbfbf9;--fg:#1b1b1b;--muted:#5b6066;--line:#d9d9d4;--card:#ffffff;--head:#efefea;--accent:#1a6fb0;--warn:#b0401a;--ok:#2e7d32}
"""
_CSS_DARK = """
:root{--bg:#111318;--fg:#e8e8e8;--muted:#B8C0CC;--line:#2a3040;--card:#1C2130;--head:#1C2130;--accent:#5BA4CF;--warn:#E05A5A;--ok:#7FC97F}
"""
_CSS = """
body{background:var(--bg);color:var(--fg);font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;line-height:1.45}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
header{background:var(--head);padding:10px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
header .brand{font-weight:600} header a{font-size:14px} header .stamp{margin-left:auto;color:var(--muted);font-size:12px}
main{max-width:1500px;margin:0 auto;padding:18px 24px 60px}
h1{font-size:26px;margin:8px 0 4px} h2{font-size:19px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:3px} h3{font-size:15px;margin:16px 0 6px;color:var(--muted)}
.lead{color:var(--muted);font-size:15px;margin:0 0 14px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px} td,th{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top;white-space:nowrap}
th{background:var(--head);cursor:pointer;position:sticky;top:46px} th.sorted-asc::after{content:" \\25B2"} th.sorted-desc::after{content:" \\25BC"}
tr:hover td{background:rgba(26,111,176,0.06)} td.num{text-align:right;font-variant-numeric:tabular-nums}
input.filter{margin:6px 0;padding:4px 8px;width:22rem;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px} .grid .card h3{margin-top:0}
figure{margin:12px 0} figure img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff} figcaption{color:var(--muted);font-size:13px;margin-top:4px}
.caveat{border-left:3px solid var(--warn);padding-left:12px;color:var(--fg)} .note{border-left:3px solid var(--accent);padding-left:12px;color:var(--muted)} .ok{border-left:3px solid var(--ok);padding-left:12px}
.missing{color:var(--warn);font-size:13px;border:1px dashed var(--warn);padding:8px;border-radius:4px}
code{background:var(--head);padding:1px 5px;border-radius:3px;font-size:12.5px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;font-size:13.5px;margin:6px 0} dl.kv dt{color:var(--muted)} dl.kv dd{margin:0}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:14px 0 0;flex-wrap:wrap} .tabs a{padding:7px 14px;border:1px solid var(--line);border-bottom:none;border-radius:6px 6px 0 0;background:var(--head);font-size:14px} .tabs a.active{background:var(--card);font-weight:600}
.slider{margin:10px 0} .slider img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff;display:block} .slider .ctl{display:flex;gap:10px;align-items:center;margin:6px 0;font-size:13px} .slider input[type=range]{flex:1} .slider button{padding:2px 10px}
svg.spark{vertical-align:middle}
.keynav{display:flex;justify-content:space-between;align-items:center;margin:4px 0;font-size:14px}
.tag{display:inline-block;background:var(--head);color:var(--muted);border-radius:4px;padding:1px 7px;font-size:12px;margin-right:6px;border:1px solid var(--line)}
"""

# ---------------------------------------------------------------------------------------- JS

_JS = r"""
// sortable, filterable tables: th onclick -> sortBy(tableId, colIndex); input oninput -> filt(tableId, text)
function sortBy(id,c){var t=document.getElementById(id),tb=t.tBodies[0],rows=Array.from(tb.rows),th=t.tHead.rows[0].cells[c];
 var asc=!th.classList.contains('sorted-asc');Array.from(t.tHead.rows[0].cells).forEach(function(h){h.classList.remove('sorted-asc','sorted-desc')});
 th.classList.add(asc?'sorted-asc':'sorted-desc');
 rows.sort(function(a,b){var x=a.cells[c].getAttribute('data-v')||a.cells[c].textContent,y=b.cells[c].getAttribute('data-v')||b.cells[c].textContent;
  var nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;return asc?x.localeCompare(y):y.localeCompare(x)});
 rows.forEach(function(r){tb.appendChild(r)})}
function filt(id,q){q=q.toLowerCase();Array.from(document.getElementById(id).tBodies[0].rows).forEach(function(r){r.style.display=r.textContent.toLowerCase().indexOf(q)>=0?'':'none'})}
// image sliders: one per .slider div, data in window.SLIDERS[id] = {src:[...], label:[...]}
window.SLIDERS=window.SLIDERS||{};
function sliderGo(id,i){var s=SLIDERS[id];i=Math.max(0,Math.min(s.src.length-1,i));s.i=i;
 document.getElementById(id+'_img').src=s.src[i];document.getElementById(id+'_lab').textContent=s.label[i];document.getElementById(id+'_rng').value=i}
function sliderStep(id,d){sliderGo(id,SLIDERS[id].i+d)}
function sliderPlay(id){var s=SLIDERS[id];if(s.timer){clearInterval(s.timer);s.timer=null;document.getElementById(id+'_play').textContent='play';return}
 document.getElementById(id+'_play').textContent='pause';s.timer=setInterval(function(){if(s.i>=s.src.length-1){sliderGo(id,0)}else{sliderStep(id,1)}},350)}
document.addEventListener('keydown',function(e){var f=document.querySelector('.slider:hover');if(!f)return;var id=f.id;
 if(e.key==='ArrowRight'){sliderStep(id,1);e.preventDefault()}if(e.key==='ArrowLeft'){sliderStep(id,-1);e.preventDefault()}});
window.addEventListener('load',function(){for(var id in SLIDERS){SLIDERS[id].src.forEach(function(u){var im=new Image();im.src=u})}});
// page-to-page keyboard navigation (keynav()): left/right arrows go to the previous/next item of the
// same group, unless the mouse is over a slider (which owns the arrows while hovered)
document.addEventListener('keydown',function(e){if(!window.KEYNAV)return;if(document.querySelector('.slider:hover'))return;
 if(e.key==='ArrowRight'&&KEYNAV.next){location.href=KEYNAV.next;e.preventDefault()}
 if(e.key==='ArrowLeft'&&KEYNAV.prev){location.href=KEYNAV.prev;e.preventDefault()}});
"""

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
PLOTLY_LOCAL_SOURCE = os.path.expanduser(
    "~/Documents/GitHub/rat_strategy_discovery/explorer/lib/plotly.min.js")


# ------------------------------------------------------------------------------------- Site

class Site:
    """One explorer site: a root folder, a title, a nav and a theme. `page()` writes a page."""

    def __init__(self, root, title, nav=(), wiki_html=None, theme="light", plotly=None):
        self.root, self.title, self.nav, self.wiki_html = root, title, list(nav), wiki_html
        self.theme, self.plotly = theme, plotly
        self.stamp = time.strftime("%Y-%m-%d %H:%M")
        self.pages = []
        os.makedirs(root, exist_ok=True)
        if plotly == "local":
            os.makedirs(os.path.join(root, "lib"), exist_ok=True)
            shutil.copy(PLOTLY_LOCAL_SOURCE, os.path.join(root, "lib", "plotly.min.js"))

    def _head(self, title, depth, extra_head=""):
        css = (_CSS_DARK if self.theme == "dark" else _CSS_LIGHT) + _CSS
        plotly = ""
        if self.plotly == "cdn":
            plotly = '<script src="%s" charset="utf-8"></script>' % PLOTLY_CDN
        elif self.plotly == "local":
            plotly = '<script src="%slib/plotly.min.js"></script>' % ("../" * depth)
        return ('<!doctype html><html><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>%s</title><style>%s</style><script>%s</script>%s%s</head>'
                % (_html.escape(title), css, _JS, plotly, extra_head))

    def _header(self, depth):
        up = "../" * depth
        links = "".join('<a href="%s%s">%s</a>' % (up, h, _html.escape(t)) for h, t in self.nav)
        wiki = ('<a href="%s">wiki</a>' % self.wiki_html) if self.wiki_html else ""
        return ('<header><span class="brand">%s</span>%s%s<span class="stamp">built %s</span></header>'
                % (_html.escape(self.title), links, wiki, self.stamp))

    def page(self, relpath, title, body, lead="", extra_head=""):
        """Write one page at root/relpath. `depth` (for relative links) is inferred from relpath."""
        depth = relpath.count("/")
        out = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        doc = (self._head(title, depth, extra_head) + "<body>" + self._header(depth) +
               '<main><h1>%s</h1>' % _html.escape(title) +
               ('<p class="lead">%s</p>' % lead if lead else "") + body + "</main></body></html>")
        with open(out, "w") as f:
            f.write(doc)
        self.pages.append(relpath)
        return relpath

    def finish(self):
        print("explorer: %d pages under %s" % (len(self.pages), self.root))


# -------------------------------------------------------------------------------- components

def esc(s):
    return _html.escape(str(s))


def table(rows, cols, tid="t", filterable=True, numeric=None, links=None):
    """Sortable, filterable table.
    rows: list of dicts; cols: list of (key, header) or keys; numeric: keys right-aligned;
    links: {key: lambda row: href} to wrap a cell in a link. Values that are floats are shown
    with 3 significant figures; None/NaN as a dash; the raw value is kept in data-v for sorting."""
    cols = [(c, c) if isinstance(c, str) else c for c in cols]
    numeric = set(numeric or [])
    links = links or {}
    th = "".join('<th onclick="sortBy(\'%s\',%d)">%s</th>' % (tid, i, esc(h)) for i, (_, h) in enumerate(cols))
    body = []
    for r in rows:
        tds = []
        for k, _ in cols:
            v = r.get(k, "")
            shown, raw = fmt(v), ("" if v is None else v)
            if k in links:
                shown = '<a href="%s">%s</a>' % (links[k](r), shown)
            tds.append('<td%s data-v="%s">%s</td>' % (' class="num"' if k in numeric else "", esc(raw), shown))
        body.append("<tr>" + "".join(tds) + "</tr>")
    filt = ('<br><input class="filter" placeholder="filter rows" oninput="filt(\'%s\',this.value)">' % tid) if filterable else ""
    return '%s<table id="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (filt, tid, th, "".join(body))


def fmt(v):
    """Display a cell value: floats to 3 significant figures, None/NaN as a dash, else escaped."""
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "&ndash;"
    except Exception:
        pass
    if isinstance(v, float):
        return "%.3g" % v if abs(v) < 1e4 else "%.0f" % v
    if hasattr(v, "__html__"):
        return v.__html__()
    return str(v) if isinstance(v, Raw) else esc(v)


class Raw(str):
    """A string that is already HTML (e.g. a sparkline) and must not be escaped by table()."""


def fig(src, caption="", root=None):
    """A figure card. If `root` is given and the file does not exist there, a visible
    'missing' box is emitted instead of a broken image, so a half-built site stays honest."""
    # normpath first: "items/../img/x.png" must resolve even when items/ does not exist yet
    # (pages are written after their bodies are composed), otherwise every relative figure on
    # an item page reports missing. Found on the first click of the first explorer.
    if root is not None and not os.path.exists(os.path.normpath(os.path.join(root, src))):
        return '<p class="missing">figure missing: %s</p>' % esc(src)
    return ('<figure><a href="%s"><img src="%s" loading="lazy"></a><figcaption>%s</figcaption></figure>'
            % (src, src, caption))


def grid(cards):
    return '<div class="grid">%s</div>' % "".join('<div class="card">%s</div>' % c for c in cards)


def card(inner, title=None):
    return '<div class="card">%s%s</div>' % (("<h3>%s</h3>" % esc(title)) if title else "", inner)


def kv(d):
    return '<dl class="kv">%s</dl>' % "".join("<dt>%s</dt><dd>%s</dd>" % (esc(k), fmt(v)) for k, v in d.items())


def note(t):
    return '<p class="note">%s</p>' % t


def caveat(t):
    return '<p class="caveat">%s</p>' % t


def ok(t):
    return '<p class="ok">%s</p>' % t


def tabs(items, active):
    """Tab bar linking sibling pages: items = [(href, label)], active = href of this page."""
    return '<div class="tabs">%s</div>' % "".join(
        '<a href="%s" class="%s">%s</a>' % (h, "active" if h == active else "", esc(l)) for h, l in items)


def keynav(prev_href, next_href, prev_label, next_label, group=""):
    """Previous/next links for flipping through the items of one group (a seed, a rat, a
    session), wired to the left and right arrow keys. Pass the wrapped-around neighbours for a
    loop: on one group the right arrow goes to the next item and the left to the previous one,
    wrapping around at the ends."""
    return ('<div class="keynav"><a href="%s">&larr; %s</a><span class="tag">%s: arrows flip</span>'
            '<a href="%s">%s &rarr;</a></div><script>window.KEYNAV={prev:%s,next:%s};</script>'
            % (prev_href, esc(prev_label), esc(group), next_href, esc(next_label),
               json.dumps(prev_href), json.dumps(next_href)))


def slider(sid, srcs, labels, caption=""):
    """An image sequence with a range input, previous/next/play buttons and arrow keys (when the
    mouse is over the slider). Images are preloaded on page load."""
    data = json.dumps({"src": list(srcs), "label": [str(l) for l in labels], "i": 0})
    n = len(srcs)
    return ('<div class="slider" id="%s"><img id="%s_img" src="%s">'
            '<div class="ctl"><button onclick="sliderStep(\'%s\',-1)">&larr;</button>'
            '<input type="range" id="%s_rng" min="0" max="%d" value="0" oninput="sliderGo(\'%s\',parseInt(this.value))">'
            '<button onclick="sliderStep(\'%s\',1)">&rarr;</button>'
            '<button id="%s_play" onclick="sliderPlay(\'%s\')">play</button>'
            '<span id="%s_lab">%s</span></div>%s'
            '<script>SLIDERS["%s"]=%s;</script></div>'
            % (sid, sid, srcs[0] if srcs else "", sid, sid, n - 1, sid, sid, sid, sid, sid,
               labels[0] if labels else "", ('<figcaption>%s</figcaption>' % caption) if caption else "",
               sid, data))


def sparkline(values, w=110, h=22, color="#1a6fb0", lo=None, hi=None, baseline=None):
    """Inline SVG line of a short series, for table cells. NaNs break the line. `baseline`
    draws a dotted reference (e.g. chance)."""
    import math
    vals = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in values]
    finite = [v for v in vals if v is not None]
    if not finite:
        return Raw("")
    lo = min(finite) if lo is None else lo
    hi = max(finite) if hi is None else hi
    if hi == lo:
        hi = lo + 1e-9
    n = len(vals)
    pts, segs = [], []
    for i, v in enumerate(vals):
        if v is None:
            if pts:
                segs.append(pts); pts = []
            continue
        x = 2 + (w - 4) * i / max(1, n - 1)
        y = 2 + (h - 4) * (1 - (v - lo) / (hi - lo))
        pts.append("%.1f,%.1f" % (x, y))
    if pts:
        segs.append(pts)
    paths = "".join('<polyline fill="none" stroke="%s" stroke-width="1.4" points="%s"/>' % (color, " ".join(s)) for s in segs)
    base = ""
    if baseline is not None and lo <= baseline <= hi:
        yb = 2 + (h - 4) * (1 - (baseline - lo) / (hi - lo))
        base = '<line x1="2" x2="%d" y1="%.1f" y2="%.1f" stroke="#999" stroke-dasharray="2,2" stroke-width="0.8"/>' % (w - 2, yb, yb)
    return Raw('<svg class="spark" width="%d" height="%d">%s%s</svg>' % (w, h, base, paths))


def plotly_page_body(payload, js, div_ids=("plot",), controls_html=""):
    """Body for a page whose figures are drawn in the browser by Plotly from an embedded
    payload. `payload` is any JSON-serialisable object, exposed as window.DATA; `js` is the
    page's own JavaScript (it may call Plotly.react on the divs). Requires Site(plotly=...)."""
    divs = "".join('<div id="%s"></div>' % d for d in div_ids)
    return ('%s%s<script>window.DATA=%s;</script><script>%s</script>'
            % (controls_html, divs, json.dumps(payload), js))


def methods_page(site, relpath, title, sections):
    """A methods page from [(heading, html)] pairs, so every explorer documents its own
    definitions where the reader can find them."""
    body = "".join("<h2>%s</h2>%s" % (esc(h), t) for h, t in sections)
    return site.page(relpath, title, body)
