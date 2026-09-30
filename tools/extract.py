"""PDF/HTML -> plain text. Text layer first (pymupdf); OCR (tesseract nep+eng) fallback per page.
Also detects legacy-font (Preeti etc.) text layers, which decode as Latin garbage, and OCRs those instead."""
import re, sys, subprocess, tempfile, os
import pymupdf; fitz = pymupdf
from bs4 import BeautifulSoup

DEV = re.compile(r'[ऀ-ॿ]')
LAT = re.compile(r'[A-Za-z]')
EN_STOP = set("the of and to in a is for with on as by are that this be or from at an it which can".split())
MAX_OCR_PAGES = int(os.environ.get("MAX_OCR_PAGES", "40"))

def script_of(text):
    d = len(DEV.findall(text)); l = len(LAT.findall(text))
    if d + l == 0: return "unknown", "unknown"
    r = d / (d + l)
    script = "devanagari" if r > .85 else "latin" if r < .10 else "mixed"
    lang = "ne" if script == "devanagari" else "en" if script == "latin" else "mixed"
    return lang, script

def looks_legacy_font(text):
    """Latin text that isn't English (Preeti/Kantipur-style mapping) -> True."""
    if len(text) < 400 or len(DEV.findall(text)) > 0.02 * len(text): return False
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 50: return False
    stop = sum(w in EN_STOP for w in words) / len(words)
    return stop < 0.06

def ocr_page(page, dpi=150):
    pix = page.get_pixmap(dpi=dpi)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.png"); pix.save(p)
        # OMP_THREAD_LIMIT=1: tesseract's own threading oversubscribes badly (minutes/page); parallelise across pages instead
        r = subprocess.run(["tesseract", p, "-", "-l", "nep+eng", "--psm", "6"],
                           capture_output=True, text=True, timeout=240, env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        return r.stdout

_DEVONLY = re.compile(r"[^\u0900-\u097F]")
def _dev(s): return _DEVONLY.sub("", clean_dev(s).replace("\u200d","").replace("\u200c",""))

def layer_score(doc, texts):
    """Char-level agreement between the PDF text layer and OCR on up to 3 text-rich pages (0..1).
    Low => text layer is mis-mapped (wrong ToUnicode / legacy fonts) and must not be trusted."""
    import difflib
    cand = sorted(range(len(texts)), key=lambda i: -len(_dev(texts[i])))[:3]
    cand = [i for i in cand if len(_dev(texts[i])) > 300]
    if not cand: return None
    sc = []
    for i in cand:
        a = _dev(texts[i])[:1500]; b = _dev(ocr_page(doc[i]))[:1500]
        if len(b) < 100: continue
        sc.append(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
    return sum(sc) / len(sc) if sc else None

_DUP = re.compile(r"([\u093E-\u094C\u0902\u0903])\1+")
def clean_dev(t):
    t = re.sub("\u094d\u094d([\u0915-\u0939])\\1?", "\u094d\\1", t)  # doubled virama + doubled consonant (PDF font artefact: कार््ययालय -> कार्यालय)
    return _DUP.sub(r"\1", t)  # doubled vowel signs (ाा, ीी, ोो) are never valid Devanagari

def extract_pdf(path):
    """returns (text, info) info: extraction, pages, ocr_pages, note"""
    doc = fitz.open(path)
    n = doc.page_count
    texts = [pg.get_text("text") for pg in doc]
    sample = "\n".join(texts[:min(n, 40)])
    legacy = looks_legacy_font(sample)
    ocr_pages = 0; note = ""; score = None
    if legacy: note = "legacy-font text layer detected; OCR used"
    elif sum(len(_dev(t)) for t in texts) > 500:
        score = layer_score(doc, texts)
        if score is not None and score < 0.90:
            legacy = True; note = f"text layer unreliable (OCR agreement {score:.2f}); OCR used"
    todo = [i for i, t in enumerate(texts) if legacy or len(t.strip()) < 40][:MAX_OCR_PAGES]
    if legacy:
        for i in range(len(texts)):
            if i not in todo: texts[i] = ""
    from concurrent.futures import ThreadPoolExecutor
    def job(i):
        try: return i, ocr_page(pymupdf.open(path)[i])
        except Exception as e: return i, None
    with ThreadPoolExecutor(3) as ex:
        for i, o in ex.map(job, todo):
            if o is None: note += f" ocr-fail p{i+1}"
            else: texts[i] = o; ocr_pages += 1
    texts = [clean_dev(t) for t in texts]
    if len(todo) == MAX_OCR_PAGES and n > MAX_OCR_PAGES: note += f" OCR capped at {MAX_OCR_PAGES} pages of {n}"
    text = "\n\n".join(f"[page {i+1}]\n{t.strip()}" for i, t in enumerate(texts) if t.strip())
    ext = "ocr" if (legacy and ocr_pages) or ocr_pages > n / 2 else "pdf-text"
    if 0 < ocr_pages <= n / 2: note += f" {ocr_pages}/{n} pages OCRed"
    return text, {"extraction": ext, "pages": n, "ocr_pages": ocr_pages, "layer_score": None if score is None else round(score, 3), "note": note.strip()}

def extract_html(html):
    s = BeautifulSoup(html, "lxml")
    for t in s(["script", "style", "nav", "header", "footer", "noscript", "form"]): t.decompose()
    main = s.find("main") or s.find("article") or s.body or s
    txt = re.sub(r"\n\s*\n+", "\n\n", main.get_text("\n", strip=True))
    return txt, {"extraction": "html", "pages": 1, "ocr_pages": 0, "note": ""}

if __name__ == "__main__":
    t, i = extract_pdf(sys.argv[1]); print(i); print(t[:1500])
