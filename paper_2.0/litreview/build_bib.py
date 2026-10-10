"""
Build paper/references.bib from authoritative metadata only (invariant I2): the arXiv API (with its journal_ref when
present) for arXiv papers, Crossref for DOIs, and fetched web pages for the two Transformer Circuits articles.
Writes ../references.bib and VERIFICATION.md (key, source URL, fetched title, first author, year) for a human check.
"""
import json
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "research-paper-pilot/1.0 (mailto:noreply@example.com)"}
ARXIV = {  # key: arXiv id
    "stolfo2024confidence": "2406.16254", "gurnee2024universal": "2401.12181", "ferrando2025iknow": "2411.14257",
    "nguyen2026commit": "2609.32964", "kadavath2022know": "2207.05221", "gekhman2024slick": "2405.05904",
    "mallen2023popqa": "2212.10511", "kalai2025hallucinate": "2509.04664", "kang2025unfamiliar": "2403.05612",
    "huh2024platonic": "2405.07987", "cheng2025abstraction": "2405.15471", "arditi2024refusal": "2406.11717",
    "rimsky2024caa": "2312.06681", "ji2025hedging": "2503.14477", "kuhn2023semantic": "2302.09664",
    "brzezinka2026familiarity": "2607.13568", "liu2026broken": "2608.15772", "lippl2024source": "2404.08101",
    "llama3": "2407.21783", "qwen25": "2412.15115", "gemma2": "2408.00118", "mistral7b": "2310.06825",
    "wang2023ioi": "2211.00593", "syed2023attribution": "2310.10348", "gurnee2026workspace": "2607.15495", "che2026steering": "2610.07270", "kumaran2026confidence": "2603.22161", "ni2025popular": "2505.17537",
}
ACL = {"teplica2025sciurus": "2025.naacl-long.618"}
# Venue records opened via the OpenReview search API on 2026-10-10 (OpenAlex had no venue location for these).
OPENREVIEW_VENUE = {
    "2411.14257": ("International Conference on Learning Representations (ICLR)", "2025", "https://openreview.net/forum?id=WCRQFlji2q", None),
    "2405.15471": ("International Conference on Learning Representations (ICLR)", "2025", "https://openreview.net/forum?id=0fD3iIBhlV", None),
    "2211.00593": ("International Conference on Learning Representations (ICLR)", "2023", "https://openreview.net/forum?id=XBflsCjpUM", None),
    "2401.12181": ("Transactions on Machine Learning Research", "2024", "https://openreview.net/forum?id=ZeI104QZ8I", None),
    "2405.07987": ("International Conference on Machine Learning (ICML)", "2024", "https://openreview.net/forum?id=BH8TYy0r6u", "Position: The Platonic Representation Hypothesis"),
}
ARXIV_SEARCH = {"teplica2025sciurus": 'ti:"SCIURus"', "olmo3": 'ti:"Olmo 3"'}
DOIS = {"reder1992": "10.1037/0278-7393.18.3.435", "schwartz1992": "10.1037/0278-7393.18.5.1074",
        "liu2007fok": "10.1080/17470210601184039", "koriat2001cascade": "10.1037/0278-7393.27.1.34"}
WEB = {"lindsey2025biology": ("https://transformer-circuits.pub/2025/attribution-graphs/biology.html", "On the Biology of a Large Language Model"),
       "anthropic2026workspace": (None, None)}

NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def get(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 5:
                time.sleep(20 * (attempt + 1))
                continue
            raise
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            if attempt < 5:
                time.sleep(10)
                continue
            raise


def abs_page(aid):
    """Metadata from the arXiv abstract page (authoritative page; the API was rate-limited on 2026-10-09)."""
    h = get(f"https://arxiv.org/abs/{aid}")
    meta = lambda n: re.findall(r'<meta name="citation_' + n + r'" content="([^"]*)"', h)
    import html as _h
    title = _h.unescape(meta("title")[0])
    authors = [_h.unescape(a) for a in meta("author")]
    date = (meta("date") or meta("online_date") or [""])[0]
    jr = re.search(r'<td class="tablecell jref">(.*?)</td>', h, re.S)
    doi = (meta("doi") or [""])[0]
    time.sleep(1.0); print("abs", aid, flush=True)
    return dict(id=aid, title=title, authors=authors, year=date[:4], jref=_h.unescape(re.sub(r"<[^>]+>", "", jr.group(1)).strip()) if jr else "", doi=doi)


def arxiv_search(q):
    h = get("https://arxiv.org/search/?searchtype=title&query=" + urllib.parse.quote(q))
    m = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', h)
    return abs_page(m.group(1)) if m else None


def arxiv_entries(query):
    root = ET.fromstring(get("https://export.arxiv.org/api/query?" + query))
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        out.append(dict(id=re.sub(r"v\d+$", "", aid), title=" ".join(e.find("a:title", NS).text.split()),
                        authors=[a.find("a:name", NS).text for a in e.findall("a:author", NS)],
                        year=e.find("a:published", NS).text[:4],
                        jref=(e.find("x:journal_ref", NS).text.strip() if e.find("x:journal_ref", NS) is not None else ""),
                        doi=(e.find("x:doi", NS).text.strip() if e.find("x:doi", NS) is not None else "")))
    return out


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def dblp_published(title):
    """Published (non-arXiv) version via OpenAlex: an exact normalised-title match with a venue location
    (a non-arXiv DOI, OpenReview, PMLR or ACL Anthology page). DBLP and Semantic Scholar blocked us on 2026-10-10."""
    try:
        j = json.loads(get("https://api.openalex.org/works?per_page=8&select=title,publication_year,locations,type&search="
                           + urllib.parse.quote(re.sub(r"[:?,]", " ", title))))
    except Exception as e:
        print("openalex failed", title[:40], e); return None
    time.sleep(0.5)
    for w in j["results"]:
        if norm(w.get("title") or "") != norm(title):
            continue
        for loc in w["locations"]:
            url = loc.get("landing_page_url") or ""
            src = (loc.get("source") or {}).get("display_name") or ""
            venue_like = (("doi.org/" in url and "10.48550" not in url) or any(d in url for d in ("openreview.net", "proceedings.mlr.press", "aclanthology.org")))
            if venue_like and "arxiv" not in src.lower() and "research online" not in src.lower() and "repository" not in src.lower():
                return dict(venue=src or url, year=str(w["publication_year"]), ee=url,
                            type="Journal Articles" if w["type"] == "article" and "journal" in src.lower() else "Conference",
                            dblp="OpenAlex: " + url)
    return None


def bib_arxiv(key, m):
    pub = m.get("pub")
    if pub and "doi.org/" in pub["ee"]:
        # published version with a DOI: take the publisher's BibTeX from Crossref, keep the arXiv id
        doi = pub["ee"].split("doi.org/", 1)[1]
        bib = get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex").strip()
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda mm: f"@{mm.group(1)}{{{key},", bib, count=1)
        cr = json.loads(get("https://api.crossref.org/works/" + doi))["message"]
        pub["venue"] = (cr.get("container-title") or [pub["venue"]])[0]
        m["authors"] = [f"{a.get('family', '')}, {a.get('given', '')}" for a in cr.get("author", [])] or m["authors"]
        return bib[:-1].rstrip() + f", eprint = {{{m['id']}}}, archivePrefix = {{arXiv}}\n}}"
    fields = {"title": "{" + m["title"] + "}", "author": " and ".join(m["authors"]), "year": m["year"],
              "eprint": m["id"], "archivePrefix": "arXiv", "url": f"https://arxiv.org/abs/{m['id']}"}
    if m.get("pub"):
        pub = m["pub"]; fields["year"] = pub["year"]
        fields["journal" if "Journal" in pub["type"] else "booktitle"] = pub["venue"]
        if pub["ee"]:
            fields["url"] = pub["ee"]
    if m["jref"]:
        fields["note"] = m["jref"]
    if m["doi"]:
        fields["doi"] = m["doi"]
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in fields.items())
    return f"@misc{{{key},\n{body}\n}}"


rows, bibs = [], []
meta = {}
for key, aid in ARXIV.items():
    try:
        meta[aid] = abs_page(aid)
    except Exception as e:
        print("failed", key, aid, e)
for aid, m in meta.items():
    if aid in OPENREVIEW_VENUE:
        v, y, u, t = OPENREVIEW_VENUE[aid]
        m["pub"] = dict(venue=v, year=y, ee=u, type="Journal Articles" if "Transactions" in v else "Conference", dblp="OpenReview: " + u)
        if t:
            m["title"] = t
    else:
        m["pub"] = dblp_published(m["title"])
for key, aid in ARXIV.items():
    m = meta.get(aid)
    if not m:
        rows.append((key, f"https://arxiv.org/abs/{aid}", "NOT FOUND", "", "")); continue
    pubtxt = f" | published: {m['pub']['venue']} {m['pub']['year']} ({m['pub']['dblp']})" if m.get("pub") else ""
    bibs.append(bib_arxiv(key, m)); rows.append((key, f"https://arxiv.org/abs/{aid}", m["title"], m["authors"][0], m["year"] + (f" | {m['jref']}" if m["jref"] else "") + pubtxt))
for key, aid in ACL.items():
    page = get(f"https://aclanthology.org/{aid}/")
    bib = get(f"https://aclanthology.org/{aid}.bib").strip()
    bib = re.sub(r"^@(\w+)\{[^,]*,", lambda mm: f"@{mm.group(1)}{{{key},", bib, count=1)
    t = re.search(r'title = "(.*?)",', bib).group(1).replace("{", "").replace("}", "")
    a = re.search(r'author = "([^,]*),', bib).group(1)
    bibs.append(bib); rows.append((key, f"https://aclanthology.org/{aid}/", t, a, "2025 | NAACL (ACL Anthology)"))
SEARCH_TITLES = {"olmo3": "Olmo 3"}
for key, q in SEARCH_TITLES.items():
    m = arxiv_search(q)
    if not m:
        rows.append((key, "arXiv title search: " + q, "NOT FOUND", "", "")); continue
    bibs.append(bib_arxiv(key, m)); rows.append((key, f"https://arxiv.org/abs/{m['id']}", m["title"], m["authors"][0], m["year"] + (f" | {m['jref']}" if m["jref"] else "")))
for key, doi in DOIS.items():
    try:
        j = json.loads(get("https://api.crossref.org/works/" + doi))["message"]
        bib = get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex").strip()
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda mm: f"@{mm.group(1)}{{{key},", bib, count=1)
        bibs.append(bib)
        rows.append((key, "https://doi.org/" + doi, j["title"][0], j["author"][0].get("family", ""), str(j["issued"]["date-parts"][0][0]) + " | " + j.get("container-title", [""])[0]))
    except Exception as e:
        rows.append((key, "https://doi.org/" + doi, f"NOT FOUND ({e})", "", ""))
# web articles: verify the page and its title
url, title = WEB["lindsey2025biology"]
html = get(url)
ok = title.lower() in html.lower()
import html as _html
txt = _html.unescape(re.sub(r"<[^>]+>", " ", html))
auth_block = txt[txt.index("Authors") + 7: txt.index("Affiliations")]
authors = [re.sub(r"[^\w .\-]", "", a).strip() for a in auth_block.split(",")]
authors = [a for a in authors if a]
year = re.search(r"Published\s+\w+ \d+, (\d{4})", txt).group(1)
bib_auth = " and ".join(a.rsplit(" ", 1)[1] + ", " + a.rsplit(" ", 1)[0] for a in authors)
bibs.append("@misc{lindsey2025biology,\n  title = {{On the Biology of a Large Language Model}},\n"
            "  author = {" + bib_auth + "},\n  year = {" + year + "},\n"
            "  howpublished = {Transformer Circuits Thread},\n  url = {" + url + "}\n}")
rows.append(("lindsey2025biology", url, title + (" (title found on page)" if ok else " (TITLE NOT FOUND ON PAGE)"), authors[0] + f" (+{len(authors)-1} authors, from page)", year))
(HERE.parent / "references.bib").write_text("\n\n".join(bibs) + "\n", encoding="utf-8")
lines = ["# Bibliography verification (built 2026-10-10 by build_bib.py from authoritative metadata)", "",
         "| key | source opened | fetched title | first author | year / venue |", "|---|---|---|---|---|"]
lines += [f"| {k} | {u} | {t} | {a} | {y} |" for k, u, t, a, y in rows]
(HERE / "VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
