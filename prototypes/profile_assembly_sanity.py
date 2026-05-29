"""Sanity check on first-pass profiles before building overlap assembly.

Tests the precondition for reference-free overlap assembly: do reads
agree on confident stretches? Absolute cumulative-mass nodes are not yet
comparable across reads (each peptide starts at a different parent
position), so the coordinate-free equivalent is recurring confident
k-mers — i.e. shared mass-difference patterns. We also look at the
profile confidence distribution, since the whole scheme rests on the
softmax probabilities being meaningful.

Usage:
    python prototypes/profile_assembly_sanity.py [antibody_substring] [k]

Defaults: antibody_substring="Herceptin", k=5.
"""

import collections
import sys

import numpy as np

NPZ_PATH = "data/pxd023419.npz"
CONF_THRESHOLD = 0.9


def load(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    P = d["profiles_concat"]
    lengths = d["lengths"].astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return {
        "P": P,
        "lengths": lengths,
        "offsets": offsets,
        "peak_file": d["peak_file"],
        "vocab": d["vocab_tokens"],
        "token_masses": d["token_masses"],
        "stop_int": int(d["stop_int"]),
    }


def main():
    antibody = sys.argv[1] if len(sys.argv) > 1 else "Herceptin"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    data = load(NPZ_PATH)
    P, lengths, offsets = data["P"], data["lengths"], data["offsets"]
    vocab, token_masses = data["vocab"], data["token_masses"]

    # Columns that are not real residues (padding, stop, zero mass).
    non_residue = set(np.where(token_masses == 0.0)[0].tolist())

    # Select one antibody's reads.
    files = np.array([f.split("/")[-1] for f in data["peak_file"]])
    sel = np.array([antibody in f for f in files])
    sel_idx = np.where(sel)[0]
    print(f"=== antibody filter: '{antibody}' ===")
    print(f"reads selected: {len(sel_idx)} / {len(lengths)}")

    # Argmax token and its confidence for every position (vectorized).
    tokens_all = P.argmax(axis=1)
    conf_all = P.max(axis=1)

    # Edge-case length stats.
    sel_len = lengths[sel_idx]
    print(
        f"profile length  min/median/max: "
        f"{int(sel_len.min())}/{int(np.median(sel_len))}/{int(sel_len.max())}"
    )
    print(
        f"  empty (len 0): {(sel_len == 0).sum()}    "
        f"hit cap (len 100): {(sel_len == 100).sum()}"
    )

    # Confidence distribution over all positions of selected reads.
    pos_mask = np.zeros(P.shape[0], dtype=bool)
    for i in sel_idx:
        pos_mask[offsets[i] : offsets[i + 1]] = True
    conf_sel = conf_all[pos_mask]
    print("\n=== top-token confidence (per position) ===")
    print(f"positions: {conf_sel.size}")
    print(f"mean: {conf_sel.mean():.3f}   median: {np.median(conf_sel):.3f}")
    for t in (0.5, 0.9, 0.99):
        print(f"  fraction >= {t}: {(conf_sel >= t).mean():.3f}")

    # Build per-read residue-label sequences in N->C order (decode is
    # C->N, so reverse). Collect confident k-mers.
    kmer_counts = collections.Counter()
    conf_label_counts = collections.Counter()  # marginal, for the null
    reads_kmers = []  # set of kmers per read, for the shareable metric

    for i in sel_idx:
        a, b = offsets[i], offsets[i + 1]
        toks = tokens_all[a:b][::-1]  # N->C
        cfs = conf_all[a:b][::-1]
        labels = [vocab[t] for t in toks]
        confident = [
            (t not in non_residue) and (c >= CONF_THRESHOLD)
            for t, c in zip(toks, cfs)
        ]
        for t, c in zip(toks, cfs):
            if t not in non_residue and c >= CONF_THRESHOLD:
                conf_label_counts[vocab[t]] += 1

        this_kmers = set()
        for j in range(len(labels) - k + 1):
            if all(confident[j : j + k]):
                km = "".join(labels[j : j + k])
                kmer_counts[km] += 1
                this_kmers.add(km)
        reads_kmers.append(this_kmers)

    total_kmers = sum(kmer_counts.values())
    n_unique = len(kmer_counts)
    print(f"\n=== confident {k}-mers (conf >= {CONF_THRESHOLD}) ===")
    print(f"total: {total_kmers}   unique: {n_unique}")
    if total_kmers == 0:
        print("no confident k-mers found; lower threshold or k")
        return

    # Null: iid draw from the confident-label marginal.
    marg_total = sum(conf_label_counts.values())
    p = {a: c / marg_total for a, c in conf_label_counts.items()}
    p_top = max(p.values())
    exp_top = total_kmers * (p_top**k)
    print(
        f"null (iid marginal): expected count of the single most likely "
        f"{k}-mer ~ {exp_top:.4f}"
    )

    print(f"\ntop 20 recurring {k}-mers (observed):")
    for km, c in kmer_counts.most_common(20):
        print(f"  {c:6d}  {km}")

    # Shareable-read metric: fraction of reads with >=1 confident k-mer
    # also seen in some other read (i.e. a candidate overlap anchor).
    recurring = {km for km, c in kmer_counts.items() if c >= 2}
    shareable = sum(
        1 for s in reads_kmers if s & recurring
    )
    n_with_kmer = sum(1 for s in reads_kmers if s)
    print("\n=== overlap precondition ===")
    print(
        f"reads with >=1 confident {k}-mer: {n_with_kmer} "
        f"({n_with_kmer / len(sel_idx):.1%} of selected)"
    )
    print(
        f"reads sharing a {k}-mer with another read: {shareable} "
        f"({shareable / len(sel_idx):.1%} of selected)"
    )
    print(
        f"recurring {k}-mers (count>=2): {len(recurring)} "
        f"({len(recurring) / n_unique:.1%} of unique)"
    )


if __name__ == "__main__":
    main()
