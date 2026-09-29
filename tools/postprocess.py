"""Post-processing pass (idempotent): clean Devanagari artefacts, recompute chars, flag agri relevance, and sort PQPMC docs into
pesticides/ (pesticide-safety material) vs text/ (other plant-quarantine docs). Run after collect.py finishes (rewrites manifest)."""
import os, re, json, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))
import extract
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAN = f"{ROOT}/data/corpus/manifest.json"
AGRI = re.compile(r"कृषि|कृषक|किसान|बाली|बालीनाली|धान|मकै|गहुँ|तरकारी|फलफूल|पशु|पशुपन्छी|बीउ|बिउ|माटो|मल\b|रासायनिक मल|खेती|विषादी|बिषादी|शत्रुजीव|कीरा|रोग|सिँचाइ|मत्स्य|माछा|मौरी|च्याउ|कफी|चिया|अलैँची|उत्पादन|agricultur|crop|livestock|pest|fertili[sz]er|seed|farm|horticult", re.I)
PEST_T = re.compile(r"विषादी|बिषादी|pesticid|पर्चा|Pamphlet|Sticker|poster", re.I)
m = json.load(open(MAN))
for r in m:
    p = f"{ROOT}/{r['text_path']}"
    if not os.path.exists(p): continue
    t = open(p).read(); t2 = extract.clean_dev(t)
    if r["source_org"] == "PQPMC":
        dest = "pesticides" if PEST_T.search(r["title"] + r["source_url"]) else "text"
        np_ = f"{ROOT}/data/corpus/{dest}/{r['id']}.txt"
        if np_ != p: os.replace(p, np_); p = np_; r["text_path"] = os.path.relpath(np_, ROOT)
        r["category"] = "pesticide-safety" if dest == "pesticides" else "plant-quarantine"
    open(p, "w").write(t2); r["chars"] = len(t2)
    hits = len(AGRI.findall(r["title"] + " " + t2[:20000])); r["agri_relevance_hits"] = hits
    r["agri_relevant"] = hits >= 5 and r["chars"] >= 300
    r["crops"] = r.get("crops", [])
json.dump(m, open(MAN + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(MAN + ".tmp", MAN)
print(len(m), "docs;", sum(r["agri_relevant"] for r in m if "agri_relevant" in r), "agri-relevant")
