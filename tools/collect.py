#!/usr/bin/env python3
"""Resumable Nepal agriculture corpus collector. Usage: python tools/collect.py [org ...]
Honours robots.txt (incl. Crawl-delay), >=1 req/s/host, retries with backoff. Never bypasses TLS/bot protection."""
import os, sys, re, json, time, hashlib, threading, datetime, urllib.robotparser
from urllib.parse import urljoin, urlparse, unquote
import requests
from bs4 import BeautifulSoup
sys.path.insert(0, os.path.dirname(__file__))
import extract

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
C = os.path.join(ROOT, "data/corpus"); RAW = f"{C}/raw"; TXT = f"{C}/text"; MAN = f"{C}/manifest.json"; FAIL = f"{C}/failures.json"
UA = "AgriChatBot-corpus/0.1 (Nepal agriculture RAG research; reason.adhikari888@gmail.com)"
MAX_BYTES = 120 * 2**20
MAX_DOCS_PER_SEED = int(os.environ.get("MAX_DOCS_PER_SEED", "150"))

# NARC/MoALD notices (vacancies, tenders) also appear in listings; kept -- filter downstream on title if desired.
# org, licence, seed urls, detail-link regex, max listing pages
SEEDS = [
 ("AITC", "nepal-gov", [f"https://aitc.gov.np/category/{c}/" for c in
   ["agricultural-diary","krshi-magazine","technical-manual","reports-and-books","other-publication","self-publication","old-publication","journal","animal-livestock-training-curriculum","procedure","directives"]], r"/content/\d+/", 6),
 ("PMAMP", "nepal-gov", [f"https://pmamp.gov.np/{p}" for p in ["publications","book","diary","directory","journal","procedures","program-and-progress","self-publishing","pocket","block","zone","superzone","technology-developed-promoted-details"]], r"technology_developed_promoted_details/|/publication|/book/", 8),
 ("MoALD", "nepal-gov", [f"https://moald.gov.np/category/{c}" for c in ["technical-publication","publication/","annual-report-of-office","statistics-related-to-agriculture","journal--magazine","other-publications","policy","guidelines-standards","procedure","study-report/","and-the-law","rules/","agricultural-development-strategy","bulletin/"]], r"/content/\d+/", 5),
 ("NARC", "nepal-gov", [f"https://narc.gov.np/{c}" for c in ["publication/factsheet","publication/agromet-bulletin","publication/brochure","publication/book-booklet","publication/proceeding","publication/annual-report","publication/newsletters","agricultural-technologies/crops","agricultural-technologies/horticulture","agricultural-technologies/livestock","agricultural-technologies/fisheries","agricultural-technologies/others","agricultural-technologies/mature-agri-technologies"]], r"/publication/content/|/agricultural-technologies/[a-z-]+/[^/]+$", 6),
]

_lock = threading.Lock(); _hostlock = {}; _last = {}; _robots = {}; _delay = {}
def load(p, d):
    try: return json.load(open(p))
    except Exception: return d
def save(p, o):
    tmp = p + ".tmp"; json.dump(o, open(tmp, "w"), ensure_ascii=False, indent=1); os.replace(tmp, p)
MANIFEST = load(MAN, []); FAILS = load(FAIL, [])
SEEN_URL = {m["source_url"] for m in MANIFEST}; SEEN_SHA = {m["sha256"] for m in MANIFEST}
FAILED_URL = {f["url"] for f in FAILS}
sess = requests.Session(); sess.headers["User-Agent"] = UA

def robots_ok(url):
    p = urlparse(url); h = p.netloc
    if h not in _robots:
        rp = urllib.robotparser.RobotFileParser(); _delay[h] = 1.0
        try:
            r = sess.get(f"{p.scheme}://{h}/robots.txt", timeout=25)
            if r.status_code == 200 and "user-agent" in r.text.lower():
                rp.parse(r.text.splitlines()); cd = rp.crawl_delay("*") or rp.crawl_delay(UA)
                if cd: _delay[h] = max(1.0, float(cd))
            else: rp.parse([])
        except Exception: rp.parse([])
        _robots[h] = rp
    return _robots[h].can_fetch(UA, url)

def get(url, stream=False):
    if not robots_ok(url): raise PermissionError("robots.txt disallows")
    h = urlparse(url).netloc
    lk = _hostlock.setdefault(h, threading.Lock())
    err = None
    for a in range(4):
        with lk:
            w = _delay.get(h, 1.0) - (time.time() - _last.get(h, 0))
            if w > 0: time.sleep(w)
            _last[h] = time.time()
        try:
            r = sess.get(url, timeout=60, stream=stream)
            if r.status_code in (429, 500, 502, 503, 504): err = f"HTTP {r.status_code}"; time.sleep(5 * 2**a); continue
            if r.status_code >= 400: raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            err = type(e).__name__ + str(e)[:80]; time.sleep(5 * 2**a)
        except requests.exceptions.SSLError as e:
            raise RuntimeError("TLS error (not bypassed): " + str(e)[:100])
    raise RuntimeError(err or "failed")

def fail(org, url, why):
    with _lock:
        if url in FAILED_URL: return
        FAILED_URL.add(url); FAILS.append({"org": org, "url": url, "reason": str(why)[:200]}); save(FAIL, FAILS)
    print("FAIL", org, url[:90], why, flush=True)

CROPS = {"धान":["धान"],"मकै":["मकै"],"गहुँ":["गहुँ","गहुं"],"आलु":["आलु"],"टमाटर":["गोलभेँडा","गोलभेडा","टमाटर"],"काउली":["काउली"],"बन्दा":["बन्दा","बन्दाकोबी"],"तोरी":["तोरी"],"उखु":["उखु"],"कफी":["कफी"],"चिया":["चिया"],"अलैंची":["अलैंची","अलैची"],"सुन्तला":["सुन्तला","सुन्तलाजात"],"स्याउ":["स्याउ"],"केरा":["केरा"],"आँप":["आँप"],"कागती":["कागती"],"किवी":["किवी"],"च्याउ":["च्याउ"],"मौरी":["मौरी","मह "],"माछा":["माछा"],"बाख्रा":["बाख्रा","बाखा"],"भैंसी":["भैंसी","भैसी"],"गाई":["गाई"],"कुखुरा":["कुखुरा"],"बंगुर":["बंगुर","वंगुर"],"जौ":["जौ "],"कोदो":["कोदो"],"दलहन":["दलहन","मसुरो","चना","भट्टमास"],"तरकारी":["तरकारी"],"अदुवा":["अदुवा"],"बेसार":["बेसार"],"लसुन":["लसुन"],"प्याज":["प्याज"],"खुर्सानी":["खुर्सानी"],"काँक्रा":["काँक्रा"],"अनार":["अनार"],"ओखर":["ओखर"]}
def crops_of(text):
    return [c for c, ks in CROPS.items() if sum(text.count(k) for k in ks) >= 3][:12]
def zone_of(text):
    z = [n for n, ks in [("terai", ["तराई", "terai", "Terai"]), ("mid-hill", ["मध्यपहाड", "मध्य पहाड", "mid-hill", "Mid-hill", "पहाडी"]), ("high-hill", ["उच्च पहाड", "हिमाली", "high hill", "High hill", "हिमाल"])] if sum(text.count(k) for k in ks) >= 3]
    return z[0] if len(z) == 1 else ("mixed" if z else "unknown") if False else (z[0] if len(z)==1 else "unknown")

def slug(s): return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()[:40] or "doc"

def add_doc(org, lic, url, title, kind="pdf", content=None, referer=None, subdir=None):
    if url in SEEN_URL or url in FAILED_URL: return False
    try:
        if content is None:
            r = get(url, stream=True)
            if int(r.headers.get("content-length", 0)) > MAX_BYTES: raise RuntimeError("too large")
            content = r.content
            if kind == "pdf" and not content[:5] == b"%PDF-": raise RuntimeError("not a PDF (" + r.headers.get("content-type", "?") + ")")
        sha = hashlib.sha256(content if isinstance(content, bytes) else content.encode()).hexdigest()
        if sha in SEEN_SHA: SEEN_URL.add(url); return False
        base = slug(unquote(os.path.basename(urlparse(url).path)) or title)
        did = f"{org.lower()}_{base}_{sha[:8]}"
        ext = "pdf" if kind == "pdf" else "html"
        rawp = f"{RAW}/{did}.{ext}"; open(rawp, "wb").write(content if isinstance(content, bytes) else content.encode())
        text, info = extract.extract_pdf(rawp) if kind == "pdf" else extract.extract_html(content)
        if len(text.strip()) < 200: raise RuntimeError(f"no usable text ({info})")
        lang, script = extract.script_of(text)
        sub = subdir or ""; os.makedirs(f"{TXT}/{sub}", exist_ok=True)
        tp = f"{TXT}/{sub}/{did}.txt" if sub else f"{TXT}/{did}.txt"
        open(tp, "w").write(text)
        rec = {"id": did, "source_url": url, "referer": referer, "source_org": org, "licence": lic, "title": title.strip()[:300],
               "language": lang, "script": script, "crops": crops_of(text), "agro_zone": zone_of(text),
               "retrieved_at": datetime.datetime.utcnow().isoformat() + "Z", "sha256": sha,
               "text_path": os.path.relpath(tp, ROOT), "chars": len(text), **info}
        with _lock:
            MANIFEST.append(rec); SEEN_URL.add(url); SEEN_SHA.add(sha); save(MAN, MANIFEST)
        print(f"OK {org} {info['extraction']:8} {len(text):8} {title[:50]}", flush=True)
        return True
    except Exception as e:
        fail(org, url, e); return False

def is_doc_link(h):
    hl = unquote(h).lower().split("?")[0]
    return hl.endswith(".pdf") or "/media/pdf_upload/" in hl

def crawl(org, lic, seeds, detail_re, maxpages):
    done = 0
    for seed in seeds:
        seen_details = set(); prev = None
        for pg in range(maxpages):
            url = seed if pg == 0 else f"{seed}{'&' if '?' in seed else '?'}page={pg}"
            try: html = get(url).text
            except Exception as e: fail(org, url, e); break
            s = BeautifulSoup(html, "lxml"); links = []
            for a in s.find_all("a", href=True):
                h = urljoin(url, a["href"].strip()); t = " ".join(a.get_text(" ", strip=True).split())
                links.append((h, t))
            sig = tuple(h for h, _ in links)
            if sig == prev: break
            prev = sig
            new_detail = []
            for h, t in links:
                if urlparse(h).netloc.endswith(".gov.np") and is_doc_link(h):
                    if done < MAX_DOCS_PER_SEED * len(seeds) and add_doc(org, lic, h, t or unquote(os.path.basename(h)), referer=url): done += 1
                elif re.search(detail_re, h) and urlparse(h).netloc == urlparse(seed).netloc and h not in seen_details and h.rstrip("/") != seed.rstrip("/"):
                    seen_details.add(h); new_detail.append((h, t))
            if pg > 0 and not new_detail and pg > 1: pass
            for h, t in new_detail:
                try: dh = get(h).text
                except Exception as e: fail(org, h, e); continue
                ds = BeautifulSoup(dh, "lxml"); title = (ds.find("h1") or ds.find("title"))
                title = title.get_text(" ", strip=True) if title else t
                got = False
                cands = [urljoin(h, a["href"].strip()) for a in ds.find_all("a", href=True)]
                cands += [u.replace(" ", "%20") for u in re.findall(r"""['"](https?://[^'"\s]+?\.pdf)['"]""", dh)]  # e.g. flipbook JS: var pdf = '...'
                for dl in dict.fromkeys(cands):
                    if urlparse(dl).netloc.endswith(".gov.np") and is_doc_link(dl):
                        if add_doc(org, lic, dl, title or t, referer=h): done += 1
                        got = True
                if not got and h not in SEEN_URL and h not in FAILED_URL:
                    txt, _ = extract.extract_html(dh)
                    if len(txt) > 2500 and org != "MoALD":
                        if add_doc(org, lic, h, title or t, kind="html", content=dh, referer=url): done += 1
    print("DONE", org, done, flush=True)

PEST = f"{C}/pesticides"
def pesticides():
    """PQPMC (npponepal.gov.np): registered/banned pesticide lists, acts, rules. Text goes to data/corpus/pesticides/."""
    os.makedirs(PEST, exist_ok=True); seen = set()
    for pg in ["https://npponepal.gov.np/", "https://npponepal.gov.np/pages/121/6696215/", "https://npponepal.gov.np/pages/75/4525666/"]:
        try: html = get(pg).text
        except Exception as e: fail("PQPMC", pg, e); continue
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            h = a["href"].strip(); t = " ".join(a.get_text(" ", strip=True).split())
            h = re.sub(r"^https?://srv5:8888/npponepal/", "https://npponepal.gov.np/", h)   # internal hostnames leaked in markup
            h = urljoin(pg, h).replace("http://npponepal", "https://npponepal")
            if urlparse(h).netloc != "npponepal.gov.np" or h in seen: continue
            if not re.search(r"\.(pdf|jpg|jpeg|png)$", unquote(h).lower()): continue
            if not re.search(r"progress|rules|notice|newdownloads", h): continue
            seen.add(h); add_pest(h, t or unquote(os.path.basename(h)), pg)
    print("DONE PQPMC", flush=True)

def add_pest(url, title, ref):
    if url in SEEN_URL or url in FAILED_URL: return
    try:
        r = get(url); content = r.content
        sha = hashlib.sha256(content).hexdigest()
        if sha in SEEN_SHA: return
        ext = os.path.splitext(urlparse(url).path)[1].lower() or ".pdf"
        did = f"pqpmc_{slug(unquote(os.path.basename(urlparse(url).path)))}_{sha[:8]}"
        rawp = f"{RAW}/{did}{ext}"; open(rawp, "wb").write(content)
        if ext == ".pdf": text, info = extract.extract_pdf(rawp)
        else:
            import subprocess
            text = subprocess.run(["tesseract", rawp, "-", "-l", "nep+eng", "--psm", "6"], capture_output=True, text=True, env=dict(os.environ, OMP_THREAD_LIMIT="1")).stdout
            info = {"extraction": "ocr", "pages": 1, "ocr_pages": 1, "note": "image (jpg) OCR"}
        if len(text.strip()) < 100: raise RuntimeError("no usable text")
        lang, script = extract.script_of(text); tp = f"{PEST}/{did}.txt"; open(tp, "w").write(text)
        rec = {"id": did, "source_url": url, "referer": ref, "source_org": "PQPMC", "licence": "nepal-gov", "title": title[:300], "language": lang, "script": script,
               "crops": [], "agro_zone": "unknown", "retrieved_at": datetime.datetime.utcnow().isoformat() + "Z", "sha256": sha,
               "text_path": os.path.relpath(tp, ROOT), "chars": len(text), "category": "pesticide-safety", **info}
        with _lock: MANIFEST.append(rec); SEEN_URL.add(url); SEEN_SHA.add(sha); save(MAN, MANIFEST)
        print(f"OK PQPMC {info['extraction']:8} {len(text):8} {title[:50]}", flush=True)
    except Exception as e: fail("PQPMC", url, e)

EXTRA = [  # (org, licence, url, title) -- hand-picked Nepal-relevant direct PDFs
 ("FAOLEX", "unknown", "https://faolex.fao.org/docs/pdf/nep159180.pdf", "Nepal green farming (FAOLEX)"),
 ("FAOLEX", "unknown", "https://faolex.fao.org/docs/pdf/nep171433.pdf", "Government of Nepal document (FAOLEX)"),
 ("FAOLEX", "unknown", "https://faolex.fao.org/docs/pdf/nep148989.pdf", "Nepal Agriculture and Food Security Country Investment Plan"),
]
def icimod():
    """ICIMOD InvenioRDM sitemap records; keep only licences that allow adaptation (no ND) and agri/Nepal relevance."""
    try: sm = get("https://lib.icimod.org/sitemap.xml").text
    except Exception as e: fail("ICIMOD", "sitemap", e); return
    for u in re.findall(r"<loc>(https://lib.icimod.org/records/[^<]+)</loc>", sm):
        try: h = get(u).text
        except Exception as e: fail("ICIMOD", u, e); continue
        title = (re.findall(r"<title>(.*?)</title>", h, re.S) or [""])[0].strip()
        lic = re.findall(r"creativecommons.org/licenses/([a-z-]+)/([0-9.]+)", h)
        if not lic: lic = [("unknown", "")]
        l = f"cc-{lic[0][0]}-{lic[0][1]}" if lic[0][0] != "unknown" else "unknown"
        if "nd" in lic[0][0].split("-")[1:]: fail("ICIMOD", u, "licence excludes derivatives (ND): skipped"); continue
        if not re.search(r"(?i)agri|crop|farm|livestock|food|soil|horticult|pastoral|rangeland|climate", title): continue
        pdf = re.findall(r'name="citation_pdf_url" content="([^"]+)"', h)
        if pdf: add_doc("ICIMOD", l, pdf[0], title, referer=u)

if __name__ == "__main__":
    for d in (RAW, TXT): os.makedirs(d, exist_ok=True)
    want = set(sys.argv[1:]); ths = []
    for org, lic, seeds, dre, mp in SEEDS:
        if want and org not in want: continue
        t = threading.Thread(target=crawl, args=(org, lic, seeds, dre, mp)); t.start(); ths.append(t)
    if not want or "EXTRA" in want:
        def ex():
            for o, l, u, t in EXTRA: add_doc(o, l, u, t)
            icimod(); print("DONE EXTRA", flush=True)
        t = threading.Thread(target=ex); t.start(); ths.append(t)
    if not want or "PQPMC" in want:
        t = threading.Thread(target=pesticides); t.start(); ths.append(t)
    for t in ths: t.join()
    print("TOTAL", len(MANIFEST))
