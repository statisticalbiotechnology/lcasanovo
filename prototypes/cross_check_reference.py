"""Validate that recurring confident k-mers map onto the known antibody.

Pure validation (uses the reference; not part of the assembly path).
Confident k-mers are recomputed as base-residue strings (modified tokens
mapped to their base AA, I->L already merged by the model) and checked as
substrings of the trastuzumab heavy (P0DOX7) and light (P0DOX8) chains.

Usage:
    python prototypes/cross_check_reference.py [antibody_substring] [k]
"""

import sys
import urllib.request

import numpy as np

NPZ_PATH = "data/pxd023419.npz"
CONF_THRESHOLD = 0.9
# Trastuzumab is a humanized IgG1/kappa. A correct reference therefore
# needs the human IgG1 heavy constant and kappa constant regions; the
# generic light-chain entries used by the pipeline (P0DOX7/P0DOX8) are
# light chains only and contain no heavy/Fc sequence.
UNIPROT = {
    "P01857": "IgG1-CH (heavy const)",
    "P01834": "IgK-C (kappa const)",
    "P0DOX7": "kappa light (generic V+C)",
    "P0DOX8": "lambda light (generic)",
}


def fetch_reference():
    seqs = {}
    for acc, name in UNIPROT.items():
        url = f"https://www.uniprot.org/uniprot/{acc}.fasta"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            lines = resp.read().decode().splitlines()
        seq = "".join(line for line in lines if not line.startswith(">"))
        # Model merges I into L; match in the same space.
        seqs[name] = seq.upper().replace("I", "L")
    return seqs


def base_letter_map(vocab, token_masses):
    """token index -> single base residue letter, or None if unmatchable."""
    mod_base = {
        "C[Carbamidomethyl]": "C",
        "M[Oxidation]": "M",
        "N[Deamidated]": "N",
        "Q[Deamidated]": "Q",
    }
    m = {}
    for idx, tok in enumerate(vocab):
        if len(tok) == 1 and tok.isalpha():
            m[idx] = "L" if tok == "I" else tok
        elif tok in mod_base:
            m[idx] = mod_base[tok]
        else:
            m[idx] = None  # pad, stop, n-term mods
    return m


def main():
    antibody = sys.argv[1] if len(sys.argv) > 1 else "Herceptin"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    d = np.load(NPZ_PATH, allow_pickle=False)
    P = d["profiles_concat"]
    lengths = d["lengths"].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    vocab = d["vocab_tokens"]
    bmap = base_letter_map(vocab, d["token_masses"])

    files = np.array([f.split("/")[-1] for f in d["peak_file"]])
    sel_idx = np.where([antibody in f for f in files])[0]

    tokens_all = P.argmax(axis=1)
    conf_all = P.max(axis=1)

    import collections

    kmer_counts = collections.Counter()
    for i in sel_idx:
        a, b = offsets[i], offsets[i + 1]
        toks = tokens_all[a:b][::-1]  # N->C
        cfs = conf_all[a:b][::-1]
        base = [bmap[t] for t in toks]
        ok = [(base[j] is not None) and (cfs[j] >= CONF_THRESHOLD)
              for j in range(len(toks))]
        for j in range(len(toks) - k + 1):
            if all(ok[j : j + k]):
                kmer_counts["".join(base[j : j + k])] += 1

    print(f"=== cross-check '{antibody}' against trastuzumab, k={k} ===")
    print(f"reads: {len(sel_idx)}   unique confident {k}-mers: {len(kmer_counts)}")

    print("fetching reference from UniProt...")
    ref = fetch_reference()
    for name, seq in ref.items():
        print(f"  {name} chain: {len(seq)} aa")
    both = list(ref.values())

    def in_ref(km):
        return any(km in s for s in both)

    # (a) Top-20 membership (which reference source contains it).
    print(f"\ntop 20 recurring {k}-mers (source?):")
    for km, c in kmer_counts.most_common(20):
        srcs = [name for name, s in ref.items() if km in s]
        print(f"  {c:6d}  {km}  -> {', '.join(srcs) if srcs else 'NO'}")

    # (b) Occurrence-weighted match rate (confident-call accuracy proxy).
    total_occ = sum(kmer_counts.values())
    matched_occ = sum(c for km, c in kmer_counts.items() if in_ref(km))
    # (c) Unique recurring match rate.
    recurring = {km: c for km, c in kmer_counts.items() if c >= 2}
    matched_uniq = sum(1 for km in recurring if in_ref(km))
    print("\n=== match rates (exact substring, I/L collapsed) ===")
    print(
        f"by occurrence: {matched_occ}/{total_occ} = "
        f"{matched_occ / total_occ:.1%}"
    )
    print(
        f"recurring (>=2) unique: {matched_uniq}/{len(recurring)} = "
        f"{matched_uniq / max(1, len(recurring)):.1%}"
    )

    # (d) Reference coverage by confident k-mers.
    print("\n=== reference coverage by confident k-mers ===")
    for name, seq in ref.items():
        covered = np.zeros(len(seq), dtype=bool)
        for km in kmer_counts:
            start = seq.find(km)
            while start != -1:
                covered[start : start + k] = True
                start = seq.find(km, start + 1)
        pct = covered.mean()
        print(f"  {name} ({len(seq)} aa): {covered.sum()} covered = {pct:.1%}")
        # Coarse 25-aa-bin coverage track to eyeball CDR dips.
        binsize = 25
        track = "".join(
            "#" if covered[j : j + binsize].mean() >= 0.5 else
            ("." if covered[j : j + binsize].mean() >= 0.1 else " ")
            for j in range(0, len(seq), binsize)
        )
        print(f"    [{track}]  (#>=50%, .>=10%, ' '<10% per 25 aa)")


if __name__ == "__main__":
    main()
