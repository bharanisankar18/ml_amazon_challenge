"""
blocking.py — Fast candidate generation for the Business Entity Resolution Challenge.

Strategy (fast, no GPU, no external lookups):
  1. Normalize name + address text (lowercase, strip legal suffixes/punctuation).
  2. Build a combined "blocking string" per record (name + address tokens).
  3. Vectorize ALL records (S1 + S2 + S3) with a single TF-IDF over character
     n-grams (robust to typos, abbreviations, transliteration variants).
  4. For each Source-1 record, use sparse cosine similarity (via sklearn's
     NearestNeighbors with metric="cosine") to pull the top-K most similar
     Source-2 and Source-3 records.
  5. Optionally widen recall with an exact/normalized-token block (sorted
     neighborhood on first name token + country) to catch near-duplicates
     that char-ngram similarity might miss.
  6. Write candidate_pairs.tsv in the required format.

Usage:
    python3 blocking.py --data-dir dataset/test --out output/candidate_pairs.tsv --topk 20

Only stdlib + pandas + scikit-learn are required (no torch/transformers),
so this step alone runs in well under a minute on tens of thousands of records.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "limited", "ltd",
    "corporation", "corp", "incorporated", "inc", "llc", "llp", "co",
    "company", "enterprises", "industries", "group", "holdings",
]

ADDR_ABBR = {
    r"\broad\b": "rd", r"\bstreet\b": "st", r"\bavenue\b": "ave",
    r"\bapartment\b": "apt", r"\bfloor\b": "fl", r"\bnear\b": "nr",
    r"\bsaint\b": "st",
}


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[.,\-_/#]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"\band\b", "&", s)
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(suf)}\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_address(s: str) -> str:
    s = normalize_text(s)
    for pat, repl in ADDR_ABBR.items():
        s = re.sub(pat, repl, s)
    return re.sub(r"\s+", " ", s).strip()


def blocking_string(name: str, addr: str) -> str:
    # Name weighted more heavily than address (repeat it) since name is the
    # stronger signal for this task; address adds disambiguating context.
    n = normalize_name(name)
    a = normalize_address(addr)
    return f"{n} {n} {a}"


# ---------------------------------------------------------------------------
# Data loading and Candidate generation
# ---------------------------------------------------------------------------

def process_source_chunked(path: str, vectorizer, is_source1=False):
    import gc
    from scipy.sparse import vstack
    
    print(f"Processing {path} ...", file=sys.stderr)
    chunks_X = []
    ids = []
    key_index = defaultdict(list)
    s1_rows = []

    def first_token_key(row_dict):
        tok = normalize_name(row_dict["business_name"]).split(" ")
        first = tok[0] if tok else ""
        return f"{row_dict.get('country','')}||{first}"

    with open(path, "r", encoding="utf-8") as f:
        # Skip header
        next(f, None)
        
        chunk_size = 10000
        current_blks = []
        current_ids = []
        
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts.extend([""] * (4 - len(parts)))
            
            entity_id, b_name, b_addr, country = parts[0], parts[1], parts[2], parts[3]
            
            blk = blocking_string(b_name, b_addr)
            current_blks.append(blk)
            current_ids.append(entity_id)
            
            row_dict = {"entity_id": entity_id, "business_name": b_name, "country": country}
            
            if is_source1:
                s1_rows.append(row_dict)
            else:
                key_index[first_token_key(row_dict)].append(entity_id)
                
            if len(current_blks) >= chunk_size:
                chunks_X.append(vectorizer.transform(current_blks))
                ids.extend(current_ids)
                current_blks = []
                current_ids = []
                
        if current_blks:
            chunks_X.append(vectorizer.transform(current_blks))
            ids.extend(current_ids)
            
    X = vstack(chunks_X) if chunks_X else None
    gc.collect()
    return X, ids, key_index, s1_rows
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder with *_source1/2/3.tsv")
    ap.add_argument("--prefix", default="test", help="File prefix, e.g. 'test' or 'train'")
    ap.add_argument("--out", required=True, help="Output path for candidate_pairs.tsv")
    ap.add_argument("--topk", type=int, default=20, help="Top-K nearest neighbors per source")
    args = ap.parse_args()

    p1 = os.path.join(args.data_dir, f"{args.prefix}_source1.tsv")
    p2 = os.path.join(args.data_dir, f"{args.prefix}_source2.tsv")
    p3 = os.path.join(args.data_dir, f"{args.prefix}_source3.tsv")

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.neighbors import NearestNeighbors
    import gc

    vectorizer = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2), n_features=100_000, alternate_sign=False, norm="l2"
    )

    X2, ids2, key_index2, _ = process_source_chunked(p2, vectorizer, is_source1=False)
    X3, ids3, key_index3, _ = process_source_chunked(p3, vectorizer, is_source1=False)
    X1, ids1, _, s1_rows = process_source_chunked(p1, vectorizer, is_source1=True)

    candidates = defaultdict(set)
    gc.collect()

    print("Running kNN search...", file=sys.stderr)
    for label, Xs, s_ids in (("S2", X2, ids2), ("S3", X3, ids3)):
        if Xs.shape[0] == 0:
            continue
        k = min(args.topk, Xs.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(Xs)
        dist, idx = nn.kneighbors(X1)
        for row_i, s1_id in enumerate(ids1):
            for col in idx[row_i]:
                candidates[s1_id].add(s_ids[col])
                
        # Free memory of X2/X3 after we are done
        del Xs, nn
        gc.collect()

    def first_token_key(row):
        tok = normalize_name(row["business_name"]).split(" ")
        first = tok[0] if tok else ""
        return f"{row.get('country','')}||{first}"

    for row in s1_rows:
        key = first_token_key(row)
        if key in key_index2:
            candidates[row["entity_id"]].update(key_index2[key])
        if key in key_index3:
            candidates[row["entity_id"]].update(key_index3[key])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ids1:
            ids = sorted(candidates.get(s1_id, []))
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    n_pairs = sum(len(v) for v in candidates.values())
    print(f"Wrote {args.out} — {len(ids1)} S1 rows, {n_pairs} total candidate pairs.", file=sys.stderr)


if __name__ == "__main__":
    main()