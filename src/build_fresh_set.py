# src/build_fresh_set.py
#
# WasteLens Iteration 9 - LOOP 6: download the PRE-REGISTERED fresh
# real-world evaluation set from Wikimedia Commons, exactly as specified in
# docs/rejection_experiment/fresh_eval_protocol.md (that document is the
# contract; this file implements it - do not change categories/counts/filters
# after results are seen).
#
# Guarantees:
#   * deterministic selection (categorymembers sorted by title, first N after
#     the registered filters; no model, no manual picks)
#   * provenance: for every file - Commons page URL, thumbnail URL, licence,
#     author, original dimensions, SHA-256 of the downloaded bytes
#   * freshness: SHA-256 of every downloaded file is checked against SHA-256
#     of every image in the training corpus (Kaggle collection) and the CIFAR
#     PNG cache; any hit is a leakage violation and is recorded loudly
#   * images go to scratch/fresh_set/ (gitignored); only the manifest,
#     metadata + hashes, is committed
#
# Run from the repo root:
#   py -3.13 -W ignore src/build_fresh_set.py            # download + manifest
#   py -3.13 -W ignore src/build_fresh_set.py --check    # verify existing set

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "WasteLensResearch/1.0 (contact: wastelens-dev@example.org)"}
OUT_DIR = Path("scratch/fresh_set")
MANIFEST = Path("docs/rejection_experiment/fresh_set_manifest.json")
CORPUS_HASH_CACHE = Path("scratch/corpus_hashes.json")
IMG_EXT = (".jpg", ".jpeg", ".png")
EXCLUDE_TITLE = ("diagram", "map", "logo", "icon", "chart", "poster", "stamp",
                 "coin", "engraving", "lithograph", "drawing", "painting",
                 "sketch", "screenshot", "book", "magazine", "manuscript",
                 "coat of arms", "graph", "blueprint", "svg")
MIN_LONG_SIDE = 400          # of the 1280px thumbnail
MAX_ASPECT = 4.0

# Pre-registered table (group, category, N, label) - verbatim from the protocol
CATEGORIES: list[tuple[str, str, int, str]] = [
    ("supported-organic", "Category:Compost", 6, "organic"),
    ("supported-organic", "Category:Food waste", 6, "organic"),
    ("supported-organic", "Category:Compost bins", 4, "organic"),
    ("supported-recyclable", "Category:Cardboard boxes", 6, "recyclable"),
    ("supported-recyclable", "Category:Aluminium cans", 5, "recyclable"),
    ("supported-recyclable", "Category:Plastic bottles", 5, "recyclable"),
    ("supported-recyclable", "Category:Broken glass", 5, "recyclable"),
    ("supported-hazardous", "Category:Lead-acid batteries", 5, "hazardous"),
    ("supported-hazardous", "Category:Button cells", 5, "hazardous"),
    ("supported-hazardous", "Category:Fluorescent lamps", 5, "hazardous"),
    ("supported-general", "Category:Litter", 6, "general trash"),
    ("supported-general", "Category:Paper towels", 5, "general trash"),
    ("ambiguous", "Category:Pizza boxes", 5, "ambiguous"),
    ("ambiguous", "Category:Cigarette butts", 5, "ambiguous"),
    ("ood-objects", "Category:Bicycles", 5, "unsupported"),
    ("ood-objects", "Category:Toy robots", 5, "unsupported"),
    ("ood-objects", "Category:Computer keyboards", 5, "unsupported"),
    ("ood-clothes", "Category:Shirts", 8, "unsupported"),
    ("ood-shoes", "Category:Shoes", 8, "unsupported"),
    ("ood-scenes", "Category:Parks", 6, "unsupported"),
    ("ood-scenes", "Category:Streets", 6, "unsupported"),
    ("multi-object", "Category:Beach litter", 6, "multi-object"),
    ("multi-object", "Category:Flea markets", 6, "multi-object"),
    ("multi-object", "Category:Clutter", 4, "multi-object"),
]


def get(url: str, timeout: int = 45) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def api(params: dict) -> dict:
    q = API + "?" + urllib.parse.urlencode(params)
    data = json.loads(get(q))
    time.sleep(1.2)                       # be polite to the Commons API
    return data


def usable_title(title: str) -> bool:
    low = title.lower()
    if not low.endswith(IMG_EXT):
        return False
    return not any(x in low for x in EXCLUDE_TITLE)


def list_category_files(category: str) -> list[str]:
    """Titles in Commons sortkey (title) order, images only, pre-filters."""
    titles: list[str] = []
    cont: dict = {}
    while True:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": category, "cmtype": "file", "cmlimit": "500",
                  "format": "json", **cont}
        data = api(params)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" in data and len(titles) < 1500:
            cont = data["continue"]
        else:
            break
    return [t for t in titles if usable_title(t)]


def imageinfo(titles: list[str]) -> dict[str, dict]:
    """Batched imageinfo (mediatype/size/thumb url) -> {title: info}."""
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        data = api({"action": "query", "titles": "|".join(chunk),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mediatype|extmetadata",
                    "iiurlwidth": "1280", "format": "json"})
        for p in data.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if ii:
                out[p["title"]] = ii
    return out


def corpus_hashes() -> set[str]:
    """SHA-256 of every image in the training/validation/test corpus."""
    if CORPUS_HASH_CACHE.exists():
        return set(json.loads(CORPUS_HASH_CACHE.read_text())["sha256"])
    import train as wl
    import train_rejection as tr
    roots = [wl.download_dataset(), tr.DATA_TMP / "cifar",
             tr.DATA_TMP / "cifar_pngs", tr.DATA_TMP / "coll"]
    hashes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    CORPUS_HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_HASH_CACHE.write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc)
         .strftime("%Y-%m-%d %H:%M:%S UTC"),
         "roots": [str(r) for r in roots], "sha256": sorted(hashes)}))
    print(f"corpus hash cache built: {len(hashes)} images")
    return hashes


def select_and_download() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    known = corpus_hashes()
    print(f"corpus reference hashes: {len(known)}")
    entries: list[dict] = []
    for group, category, n_want, label in CATEGORIES:
        titles = list_category_files(category)
        info = imageinfo(titles)
        picked = []
        for t in titles:                          # already title-sorted
            ii = info.get(t)
            if not ii or ii.get("mediatype") != "BITMAP":
                continue
            w = ii.get("thumbwidth") or ii.get("width")
            h = ii.get("thumbheight") or ii.get("height")
            if not w or not h:
                continue
            if max(w, h) < MIN_LONG_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
                continue
            picked.append((t, ii))
            if len(picked) >= n_want:
                break
        print(f"  {category:36s} want {n_want} -> selected {len(picked)} "
              f"(of {len(titles)} filtered candidates)")
        for rank, (t, ii) in enumerate(picked):
            slug = category.replace("Category:", "").replace(" ", "_")
            ext = ".png" if t.lower().endswith(".png") else ".jpg"
            dest = OUT_DIR / f"{group}__{slug}__{rank:02d}{ext}"
            url = ii.get("thumburl") or ii["url"]
            if not dest.exists():
                dest.write_bytes(get(url))
                time.sleep(0.5)                   # pace thumbnail downloads
            raw = dest.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            meta = ii.get("extmetadata", {})
            entries.append({
                "group": group, "label": label, "category": category,
                "file_title": t, "local_name": dest.name,
                "sha256": sha, "bytes": len(raw),
                "page": "https://commons.wikimedia.org/wiki/"
                        + urllib.parse.quote(t.replace(" ", "_")),
                "thumb_url": url, "original_url": ii.get("url"),
                "license": (meta.get("LicenseShortName", {}) or {}).get("value"),
                "author": (meta.get("Artist", {}) or {}).get("value", "")[:160],
                "orig_width": ii.get("width"), "orig_height": ii.get("height"),
                "thumb_width": ii.get("thumbwidth"),
                "thumb_height": ii.get("thumbheight"),
                "sha256_in_training_corpus": sha in known,
            })
    leaks = [e for e in entries if e["sha256_in_training_corpus"]]
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    manifest = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/fresh_eval_protocol.md",
        "source": "Wikimedia Commons (per-file page + licence recorded)",
        "n_images": len(entries), "counts_by_group": counts,
        "expected_counts": {"supported": 63, "ambiguous": 10, "ood": 43,
                            "multi-object": 16},
        "corpus_reference_hashes": len(known),
        "dedup_matches": [e["local_name"] for e in leaks],
        "fresh": len(leaks) == 0,
        "entries": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"selected {len(entries)} images -> {OUT_DIR}")
    print(f"dedup_matches: {manifest['dedup_matches']}  fresh={manifest['fresh']}")
    print(f"wrote {MANIFEST}")
    return manifest


def check_only() -> None:
    m = json.loads(MANIFEST.read_text())
    missing = [e["local_name"] for e in m["entries"]
               if not (OUT_DIR / e["local_name"]).exists()]
    bad = [e["local_name"] for e in m["entries"]
           if (OUT_DIR / e["local_name"]).exists()
           and hashlib.sha256(
               (OUT_DIR / e["local_name"]).read_bytes()).hexdigest()
           != e["sha256"]]
    print(f"manifest: {m['n_images']} images, fresh={m['fresh']}, "
          f"dedup_matches={m['dedup_matches']}")
    print(f"counts: {m['counts_by_group']}")
    print(f"missing files: {missing}")
    print(f"hash mismatches: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify an existing download against the manifest")
    args = ap.parse_args()
    check_only() if args.check else select_and_download()


if __name__ == "__main__":
    main()
