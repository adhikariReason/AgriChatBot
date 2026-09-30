"""Re-extract given manifest ids from data/corpus/raw with a higher OCR page cap (e.g. the pesticide registers).
Usage: MAX_OCR_PAGES=400 python tools/reextract.py <id> [<id> ...]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import extract
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAN = f"{ROOT}/data/corpus/manifest.json"
m = json.load(open(MAN))
for r in m:
    if r["id"] in sys.argv[1:]:
        raw = f"{ROOT}/data/corpus/raw/{r['id']}.pdf"
        text, info = extract.extract_pdf(raw)
        open(f"{ROOT}/{r['text_path']}", "w").write(text)
        r.update(info); r["chars"] = len(text); print(r["id"], info)
json.dump(m, open(MAN + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(MAN + ".tmp", MAN)
