# src/build_adaptation_set.py
#
# WasteLens Iteration 10 - LOOP 5/6: download the PRE-REGISTERED
# fresh-domain adaptation set from Wikimedia Commons, exactly as specified
# in docs/rejection_experiment/iteration10_adaptation_protocol.md §1.
#
# Non-contamination (protocol §0): same categories as the fresh set, but
# NEW files only - every fresh title is skipped, a rank offset of 50 past
# every fresh selection is applied, and every download is SHA-256 checked
# against all 131 fresh files AND the training corpus + CIFAR PNG cache.
# Images go to scratch/adaptation_set/ (gitignored); only the manifest is
# committed.
#
# Run from the repo root:
#   py -3.13 -W ignore src/build_adaptation_set.py
#   py -3.13 -W ignore src/build_adaptation_set.py --check

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from build_fresh_set import (IMG_EXT, corpus_hashes, get, imageinfo,
                             list_category_files, usable_title)

OUT_DIR = Path("scratch/adaptation_set")
MANIFEST = Path("docs/rejection_experiment/adaptation_set_manifest.json")
FRESH_MANIFEST = Path("docs/rejection_experiment/fresh_set_manifest.json")
RANK_OFFSET = 50

# (group, category, N, label) - §1 of iteration10_adaptation_protocol.md.
CATEGORIES: list[tuple[str, str, int, str]] = [
    ("adapt-supported", "Category:Food waste", 8, "organic"),
    ("adapt-supported", "Category:Compost", 8, "organic"),
    ("adapt-supported", "Category:Cardboard boxes", 8, "recyclable"),
    ("adapt-supported", "Category:Plastic bottles", 8, "recyclable"),
    ("adapt-supported", "Category:Lead-acid batteries", 6, "hazardous"),
    ("adapt-supported", "Category:Button cells", 6, "hazardous"),
    ("adapt-supported", "Category:Litter", 8, "general trash"),
    ("adapt-supported", "Category:Paper towels", 8, "general trash"),
    ("adapt-ood", "Category:Toy robots", 6, "unsupported"),
    ("adapt-ood", "Category:Computer keyboards", 6, "unsupported"),
    ("adapt-ood", "Category:Shirts", 6, "unsupported"),
    ("adapt-ood", "Category:Shoes", 6, "unsupported"),
    ("adapt-ood", "Category:Parks", 4, "unsupported"),
    ("adapt-ood", "Category:Streets", 4, "unsupported"),
    ("adapt-multi", "Category:Beach litter", 6, "unsupported"),
    ("adapt-multi", "Category:Flea markets", 6, "unsupported"),
]


def select_and_download() -> dict:
    fresh = json.loads(FRESH_MANIFEST.read_text(encoding="utf-8"))
    fresh_titles = {e["file_title"] for e in fresh["entries"]}
    fresh_hashes = {e["sha256"] for e in fresh["entries"]}
    known = corpus_hashes()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    skipped_fresh_overlap = 0
    for group, category, n, label in CATEGORIES:
        titles = sorted(set(list_category_files(category)))
        ii = imageinfo(titles)
        cands = [(t, ii[t]) for t in titles
                 if t in ii and t.lower().endswith(IMG_EXT)
                 and usable_title(t)]
        new_cands = [c for c in cands if c[0] not in fresh_titles]
        skipped_fresh_overlap += len(cands) - len(new_cands)
        pool = new_cands[RANK_OFFSET:RANK_OFFSET + n]
        if len(pool) < n:
            print(f"      SHORTFALL {category}: have {len(pool)}, want {n}")
        picked = sorted(pool, key=lambda c: c[0])
        for rank, (t, ii) in enumerate(picked):
            slug = category.replace("Category:", "").replace(" ", "_")
            ext = ".png" if t.lower().endswith(".png") else ".jpg"
            dest = OUT_DIR / f"{group}__{slug}__{rank:02d}{ext}"
            url = ii.get("thumburl") or ii["url"]
            if not dest.exists():
                dest.write_bytes(get(url))
                time.sleep(0.5)
            raw = dest.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha in fresh_hashes:
                raise SystemExit(f"LEAK: {dest.name} matches held-out fresh")
            meta = ii.get("extmetadata", {})
            entries.append({
                "group": group, "label": label, "category": category,
                "file_title": t, "local_name": dest.name,
                "sha256": sha, "bytes": len(raw),
                "page": ("https://commons.wikimedia.org/wiki/"
                         + urllib.parse.quote(t.replace(" ", "_"))),
                "thumb_url": url, "original_url": ii.get("url"),
                "license": (meta.get("LicenseShortName", {}) or {})
                           .get("value"),
                "author": (meta.get("Artist", {}) or {})
                          .get("value", "")[:160],
                "orig_width": ii.get("width"),
                "orig_height": ii.get("height"),
                "thumb_width": ii.get("thumbwidth"),
                "thumb_height": ii.get("thumbheight"),
                "sha256_in_training_corpus": sha in known,
                "sha256_in_fresh_set": sha in fresh_hashes,
            })
    leaks = [e for e in entries if e["sha256_in_training_corpus"]]
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    manifest = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/"
                    "iteration10_adaptation_protocol.md",
        "source": "Wikimedia Commons (per-file page + licence recorded)",
        "n_images": len(entries), "counts_by_group": counts,
        "expected_counts": {"adapt-supported": 60, "adapt-ood": 32,
                            "adapt-multi": 12},
        "rank_offset": RANK_OFFSET,
        "fresh_titles_skipped": skipped_fresh_overlap,
        "corpus_reference_hashes": len(known),
        "fresh_reference_hashes": len(fresh_hashes),
        "dedup_matches_corpus": [e["local_name"] for e in leaks],
        "fresh": len(leaks) == 0,
        "entries": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n",
                        encoding="utf-8")
    print(f"selected {len(entries)} images -> {OUT_DIR}")
    print(f"fresh titles skipped: {skipped_fresh_overlap}")
    print(f"dedup_matches_corpus: {manifest['dedup_matches_corpus']}  "
          f"fresh={manifest['fresh']}")
    print(f"wrote {MANIFEST}")
    return manifest


def check_only() -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [e["local_name"] for e in m["entries"]
               if not (OUT_DIR / e["local_name"]).exists()]
    bad = [e["local_name"] for e in m["entries"]
           if (OUT_DIR / e["local_name"]).exists()
           and hashlib.sha256(
               (OUT_DIR / e["local_name"]).read_bytes()).hexdigest()
           != e["sha256"]]
    print(f"manifest: {m['n_images']} images, fresh={m['fresh']}, "
          f"dedup_matches_corpus={m['dedup_matches_corpus']}")
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
