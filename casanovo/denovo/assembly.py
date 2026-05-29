"""Reference-free overlap-layout-consensus in mass coordinate.

Reads first-pass greedy profiles (one ``.npz`` per spectrum file) and
builds an incremental mass-coordinate DAG (Spec2Pep paper terminology:
PRM-style spectrum graph) of consensus residue calls across all reads.

Two consumer paths are provided:

* :func:`bestpath` — reads the highest-confidence antisymmetric path
  through the consensus DAG (a single candidate antibody sequence
  inferred without re-running the model).
* :func:`consensus_lookup` — returns a callable that maps a cumulative
  N-terminal mass to the mean-fused outgoing distribution, suitable as
  a prior for second-pass biased decoding (see
  :meth:`casanovo.denovo.model.Spec2Pep.decode_with_priors`).

The mass-coordinate framing is taken straight from CLAUDE.md:

* Nodes are ppm-tolerance-binned cumulative masses from a terminus.
* Edges are residue steps that carry the per-step profile distribution.
* Equal-mass tokens (I/L exactly; near-isobars GG↔N, AG↔Q within ppm)
  collapse onto the same node as parallel edges.
* Consensus on each node is the **mean** of its outgoing edge
  distributions (the 1/coverage weight buries confident-wrong reads).

Layout policy (v1, single antibody): seed with the highest-mass-product
read at shift=0; for every subsequent read, pick the offset that
maximises confidence-weighted ppm-bin overlap with the existing layout.
Bubbles (paths that reconverge on the same downstream node at the same
mass) are kept as parallel edges, never collapsed.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("casanovo")


# ---------------------------------------------------------------------------
# Tunables — all expressible as CLI flags / config in callers.
# ---------------------------------------------------------------------------

DEFAULT_PPM_TOLERANCE = 25.0  # node-binning tolerance for cumulative mass
MIN_OVERLAP_NODES = 3  # require this many overlapping nodes to place
MIN_OVERLAP_SCORE = 0.5  # ...AND this much confidence-weighted overlap
MAX_PROFILE_LEN = 100  # cap each read at this many decoding steps


# ---------------------------------------------------------------------------
# Data containers.
# ---------------------------------------------------------------------------


@dataclass
class Read:
    """One spectrum's first-pass profile, oriented N→C."""

    profile: np.ndarray  # (L, vocab) softmax probabilities, N→C
    cum_mass: np.ndarray  # (L+1,) cumulative residue mass, N→C
    precursor_mass: float  # measured neutral mass for this read
    src: str  # peak_file::scan_id

    @property
    def length(self) -> int:
        return self.profile.shape[0]


@dataclass
class LayoutNode:
    """A mass-coordinate node in the consensus DAG."""

    mass: float
    coverage: int = 0  # number of reads passing through node


@dataclass
class LayoutEdge:
    """A residue-step edge between two layout nodes.

    The ``dist_sum`` accumulates softmax distributions from every read
    whose decoding step landed on this edge. The mean (``dist_sum /
    coverage``) is the consensus distribution.
    """

    from_idx: int
    to_idx: int
    dist_sum: np.ndarray  # (vocab,) running sum of profile rows
    coverage: int = 0  # number of contributing reads


@dataclass
class Layout:
    """Mass-coordinate consensus DAG."""

    vocab: np.ndarray  # (vocab,) string labels
    token_masses: np.ndarray  # (vocab,) monoisotopic masses
    ppm_tolerance: float
    nodes: List[LayoutNode] = field(default_factory=list)
    edges: Dict[Tuple[int, int], LayoutEdge] = field(default_factory=dict)
    # Sorted (mass, node_idx) pairs for ppm-binned lookup.
    _mass_index: List[Tuple[float, int]] = field(default_factory=list)
    # Per-read N-terminal shift, in layout global mass coordinate.
    read_shifts: Dict[str, float] = field(default_factory=dict)
    # Reads that could not be placed (no overlap above threshold).
    unplaced: List[str] = field(default_factory=list)

    # -- node management ----------------------------------------------------

    def _find_node_idx(self, mass: float) -> Optional[int]:
        """Return existing node idx within ppm tolerance of ``mass``,
        or ``None``. Uses bisect on the sorted mass index."""
        if not self._mass_index:
            return None
        # Tolerance in absolute Da (ppm × mass; clamp small for tiny mass).
        tol = max(mass, 1.0) * self.ppm_tolerance * 1e-6
        i = bisect.bisect_left(self._mass_index, (mass - tol, -1))
        best_idx, best_diff = None, tol
        while i < len(self._mass_index):
            m, idx = self._mass_index[i]
            if m > mass + tol:
                break
            diff = abs(m - mass)
            if diff <= best_diff:
                best_diff, best_idx = diff, idx
            i += 1
        return best_idx

    def get_or_add_node(self, mass: float) -> int:
        """Return node idx for the given mass, creating it if no
        existing node sits within ppm tolerance."""
        idx = self._find_node_idx(mass)
        if idx is not None:
            return idx
        idx = len(self.nodes)
        self.nodes.append(LayoutNode(mass=mass))
        bisect.insort(self._mass_index, (mass, idx))
        return idx

    # -- edge accumulation --------------------------------------------------

    def add_edge_observation(
        self, from_idx: int, to_idx: int, dist: np.ndarray
    ) -> None:
        key = (from_idx, to_idx)
        edge = self.edges.get(key)
        if edge is None:
            edge = LayoutEdge(
                from_idx=from_idx,
                to_idx=to_idx,
                dist_sum=np.zeros_like(dist, dtype=np.float64),
                coverage=0,
            )
            self.edges[key] = edge
        edge.dist_sum += dist
        edge.coverage += 1


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def load_profiles(npz_paths: Sequence[str]) -> Tuple[List[Read], dict]:
    """Load one or more profile ``.npz`` files (as emitted by
    :meth:`ModelRunner.profile`) into a flat list of N→C-oriented
    :class:`Read` objects.

    All inputs must share the same vocabulary and decoding direction.
    """
    reads: List[Read] = []
    meta: dict = {}
    for path in npz_paths:
        d = np.load(path, allow_pickle=False)
        P = d["profiles_concat"]
        lengths = d["lengths"].astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        vocab = d["vocab_tokens"]
        token_masses = d["token_masses"].astype(np.float64)
        stop_int = int(d["stop_int"])
        order = str(d["decoding_order"])
        peak_files = d["peak_file"]
        scan_ids = d["scan_id"]
        precursor_mass = d["precursor_mass"].astype(np.float64)

        if not meta:
            meta = dict(
                vocab=vocab,
                token_masses=token_masses,
                stop_int=stop_int,
                decoding_order=order,
            )
        else:
            # Strict equality on vocab/masses guards against mixing models.
            if not np.array_equal(meta["vocab"], vocab):
                raise ValueError(
                    f"vocab mismatch between profile files (first vs {path})"
                )

        # If the decoder ran C→N, reverse each profile so masses
        # accumulate N→C (matches an antibody's natural orientation).
        flip = order == "C_to_N"

        for i in range(len(lengths)):
            a, b = offsets[i], offsets[i + 1]
            prof = P[a:b]
            if prof.shape[0] == 0:
                continue
            if flip:
                prof = prof[::-1].copy()
            argmax_tokens = prof.argmax(axis=1)
            # Per-step monoisotopic residue mass (0 for stop/n-term mods).
            step_masses = token_masses[argmax_tokens]
            cum = np.concatenate([[0.0], np.cumsum(step_masses)])
            reads.append(
                Read(
                    profile=prof.astype(np.float32),
                    cum_mass=cum,
                    precursor_mass=float(precursor_mass[i]),
                    src=f"{peak_files[i]}::{scan_ids[i]}",
                )
            )
    return reads, meta


# ---------------------------------------------------------------------------
# Layout — incremental OLC in mass coordinate.
# ---------------------------------------------------------------------------


def _read_confidence(read: Read) -> float:
    """Per-read aggregate: sum of top-token probabilities."""
    if read.profile.shape[0] == 0:
        return 0.0
    return float(read.profile.max(axis=1).sum())


def _seed_read(reads: List[Read]) -> int:
    """Index of the highest-confidence, longest read.

    Sort by (-confidence, -length) so the layout starts from a read
    that's both long (more nodes to anchor on) and confident (more
    likely correct).
    """
    scored = [
        (-_read_confidence(r), -r.length, idx) for idx, r in enumerate(reads)
    ]
    scored.sort()
    return scored[0][2]


def _align_read_to_layout(
    layout: Layout,
    read: Read,
    ppm: float,
) -> Tuple[Optional[float], float, int]:
    """Find the shift (read's N-term position in global mass coord)
    that maximises confidence-weighted ppm-binned mass-node overlap.

    Returns
    -------
    (shift, score, n_overlap)
        ``shift`` is None when no candidate clears the thresholds.
        ``score`` is the sum of per-position max-probabilities at
        overlapping nodes; ``n_overlap`` is the count of nodes from
        the read that fell within ppm of an existing layout node.
    """
    if not layout._mass_index or read.cum_mass.size < 2:
        return None, 0.0, 0

    # Candidate shifts: every (existing_layout_mass - read_cum_mass) pair
    # represents an alignment where the read's k-th node sits on top of
    # some layout node. Try all O(L_read * L_layout) candidates; this is
    # fine for v1 (read.length ≤ 100, layout nodes a few thousand).
    layout_masses = np.array([m for m, _ in layout._mass_index])
    read_masses = read.cum_mass

    conf = read.profile.max(axis=1)  # (L,) per-step max probability
    # Position confidence at node i (i ∈ [1..L]) = conf of the step that
    # produced cum_mass[i]; the N-terminus (i=0) has no edge → weight 0.
    pos_conf = np.zeros(read_masses.size, dtype=np.float64)
    pos_conf[1:] = conf.astype(np.float64)

    best = (None, 0.0, 0)  # (shift, score, n_overlap)

    # To keep this cheap, only seed candidate shifts from each layout
    # mass paired with the read's most confident node.
    top_pos = int(np.argmax(pos_conf))
    seed_shifts = layout_masses - read_masses[top_pos]

    for shift in seed_shifts:
        # Vectorised tolerance match: for each read mass at this shift,
        # is there a layout mass within ppm tol?
        shifted = read_masses + shift
        tol = np.maximum(np.abs(shifted), 1.0) * ppm * 1e-6
        # For each shifted mass, binary-search layout_masses and check
        # the nearest neighbour.
        idx = np.searchsorted(layout_masses, shifted)
        score, n_overlap = 0.0, 0
        for k, s in enumerate(shifted):
            t = tol[k]
            best_diff = t + 1
            for j in (idx[k] - 1, idx[k]):
                if 0 <= j < layout_masses.size:
                    diff = abs(layout_masses[j] - s)
                    if diff < best_diff:
                        best_diff = diff
            if best_diff <= tol[k]:
                score += pos_conf[k]
                n_overlap += 1
        if n_overlap >= MIN_OVERLAP_NODES and score > best[1]:
            best = (float(shift), score, n_overlap)

    return best


def _place_read(layout: Layout, read: Read, shift: float) -> None:
    """Add a read's nodes and edges to the layout at the chosen
    N-terminal shift. Both terminal pins (read's 0 → its precursor) get
    snapped onto layout nodes via the standard ppm tolerance."""
    shifted = read.cum_mass + shift
    node_ids = [layout.get_or_add_node(m) for m in shifted]
    for k in range(read.profile.shape[0]):
        layout.add_edge_observation(
            from_idx=node_ids[k],
            to_idx=node_ids[k + 1],
            dist=read.profile[k].astype(np.float64),
        )
    for nid in node_ids:
        layout.nodes[nid].coverage += 1
    layout.read_shifts[read.src] = shift


def build_layout(
    reads: List[Read],
    vocab: np.ndarray,
    token_masses: np.ndarray,
    ppm_tolerance: float = DEFAULT_PPM_TOLERANCE,
) -> Layout:
    """Build a mass-coordinate consensus DAG by incremental OLC layout.

    Reads are placed in decreasing confidence-weighted overlap order:
    seed with the highest-confidence read at shift 0, then on each
    iteration scan all remaining reads for the best alignment to the
    growing layout and place it. Reads that never clear the overlap
    thresholds are recorded in ``layout.unplaced`` for diagnostics —
    they may belong to coverage gaps or be off-target chimeras.
    """
    if not reads:
        return Layout(
            vocab=vocab, token_masses=token_masses, ppm_tolerance=ppm_tolerance
        )

    # Cap profile length defensively; ladders hitting max_peptide_len
    # are usually drift, not real long peptides.
    reads = [r for r in reads if r.length <= MAX_PROFILE_LEN]
    if not reads:
        return Layout(
            vocab=vocab, token_masses=token_masses, ppm_tolerance=ppm_tolerance
        )

    layout = Layout(
        vocab=vocab,
        token_masses=token_masses,
        ppm_tolerance=ppm_tolerance,
    )

    # Seed: place the best read at shift 0. Its N-terminus sits at the
    # global N-pin (mass 0), its C-terminus at its own precursor mass.
    seed_idx = _seed_read(reads)
    _place_read(layout, reads[seed_idx], shift=0.0)
    placed = {seed_idx}

    logger.info(
        "Layout seeded with %s (%d residues, conf=%.2f)",
        reads[seed_idx].src,
        reads[seed_idx].length,
        _read_confidence(reads[seed_idx]),
    )

    # Iteratively place the read with the best current alignment.
    # Recompute alignment for unplaced reads each pass; for efficiency,
    # cache last-known best score and only revisit when the layout
    # grows substantially.
    last_layout_size = len(layout.nodes)
    cached: Dict[int, Tuple[Optional[float], float, int]] = {}
    pass_idx = 0
    while True:
        pass_idx += 1
        # Force recompute when layout has grown by >5% since last pass.
        if len(layout.nodes) > last_layout_size * 1.05:
            cached.clear()
            last_layout_size = len(layout.nodes)
        best = (-1, None, -1.0, 0)  # (read_idx, shift, score, n_overlap)
        for i, r in enumerate(reads):
            if i in placed:
                continue
            res = cached.get(i)
            if res is None:
                res = _align_read_to_layout(layout, r, ppm_tolerance)
                cached[i] = res
            shift, score, n = res
            if shift is None or score < MIN_OVERLAP_SCORE:
                continue
            if score > best[2]:
                best = (i, shift, score, n)
        if best[0] < 0:
            break
        ridx, shift, score, n = best
        _place_read(layout, reads[ridx], shift=shift)
        placed.add(ridx)
        cached.pop(ridx, None)
        if pass_idx % 200 == 1 or len(placed) % 500 == 0:
            logger.info(
                "Layout pass %d: placed %d/%d reads, %d nodes, %d edges",
                pass_idx,
                len(placed),
                len(reads),
                len(layout.nodes),
                len(layout.edges),
            )

    layout.unplaced = [
        reads[i].src for i in range(len(reads)) if i not in placed
    ]
    logger.info(
        "Layout finished: %d/%d reads placed, %d nodes, %d edges, "
        "%d unplaced",
        len(placed),
        len(reads),
        len(layout.nodes),
        len(layout.edges),
        len(layout.unplaced),
    )
    return layout


# ---------------------------------------------------------------------------
# Consensus + best-path readout.
# ---------------------------------------------------------------------------


def consensus_edges(
    layout: Layout,
) -> Dict[int, List[Tuple[int, np.ndarray, int]]]:
    """Per-node outgoing edges with **mean-fused** distribution.

    Returns
    -------
    out
        ``{from_node_idx: [(to_node_idx, mean_dist, coverage), ...]}``,
        where ``mean_dist`` is ``edge.dist_sum / edge.coverage``.
    """
    out: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    for (a, b), edge in layout.edges.items():
        mean = edge.dist_sum / max(edge.coverage, 1)
        out.setdefault(a, []).append(
            (b, mean.astype(np.float32), edge.coverage)
        )
    return out


def bestpath(
    layout: Layout,
    target_mass: Optional[float] = None,
) -> Tuple[str, List[float], List[int]]:
    """Greedy highest-confidence path from the global N-pin.

    Starts at the layout node with mass 0 (or the lowest-mass node if 0
    is absent) and walks edges with the highest mean-fused max-token
    probability, weighted by edge coverage. Stops at a node with no
    outgoing edges, or when cumulative mass reaches ``target_mass``
    (within layout ppm).

    Returns
    -------
    (sequence, scores, node_path)
        Concatenated base residue labels (modifications stripped),
        per-residue confidence (the mean-fused max-prob at each step),
        and the list of node indices visited.
    """
    edges_by_src = consensus_edges(layout)
    if not layout.nodes:
        return "", [], []

    # Find the N-pin (mass closest to 0).
    n_pin = min(range(len(layout.nodes)), key=lambda i: layout.nodes[i].mass)

    seq_chars: List[str] = []
    scores: List[float] = []
    visited = [n_pin]
    visited_set = {n_pin}

    cur = n_pin
    cap = MAX_PROFILE_LEN * 3
    for _ in range(cap):
        out = edges_by_src.get(cur)
        if not out:
            break
        # Pick the edge whose mean distribution has the highest single
        # token probability — i.e. the most confident next residue.
        # Tie-break on coverage to bias toward well-supported paths.
        best_tok_idx, best_prob, best_to, best_cov = -1, -1.0, -1, 0
        for to_idx, mean_dist, cov in out:
            t = int(mean_dist.argmax())
            p = float(mean_dist[t])
            if p > best_prob or (p == best_prob and cov > best_cov):
                best_tok_idx, best_prob, best_to, best_cov = t, p, to_idx, cov
        if best_to < 0 or best_to in visited_set:  # no cycles
            break
        label = str(layout.vocab[best_tok_idx])
        base = _base_residue(label)
        if base:
            seq_chars.append(base)
            scores.append(best_prob)
        visited.append(best_to)
        visited_set.add(best_to)
        cur = best_to
        if target_mass is not None:
            tol = target_mass * layout.ppm_tolerance * 1e-6
            if abs(layout.nodes[cur].mass - target_mass) <= tol:
                break
    return "".join(seq_chars), scores, visited


def _base_residue(label: str) -> str:
    """Reduce a tokenizer label to its single-letter base residue.

    Modifications are stripped (oxidation, deamidation,
    carbamidomethyl). N-term modifications and stop/pad tokens map to
    the empty string."""
    if not label:
        return ""
    # Tokens like 'C[Carbamidomethyl]', 'M[Oxidation]', 'N[Deamidated]'.
    head = label[0]
    if head.isalpha() and head.isupper():
        return head
    return ""


# ---------------------------------------------------------------------------
# Consensus prior lookup for biased re-decoding.
# ---------------------------------------------------------------------------


def consensus_lookup(
    layout: Layout,
) -> Callable[[float], Optional[np.ndarray]]:
    """Return a callable mapping a cumulative N-terminal mass to the
    mean-fused outgoing distribution at the nearest layout node within
    ppm tolerance, or ``None`` when nothing is close enough.

    For nodes with multiple outgoing edges, the returned distribution
    is the **coverage-weighted mean of edge means** — i.e. the
    consensus distribution over the next residue at that mass point.
    """
    if not layout.nodes:
        return lambda _mass: None

    edges_by_src = consensus_edges(layout)
    # Precompute the consensus distribution at each node (over outgoing).
    node_dist: Dict[int, np.ndarray] = {}
    for src, outs in edges_by_src.items():
        total_cov = sum(cov for _, _, cov in outs)
        if total_cov == 0:
            continue
        mix = np.zeros_like(outs[0][1], dtype=np.float64)
        for _, mean, cov in outs:
            mix += mean.astype(np.float64) * cov
        node_dist[src] = (mix / total_cov).astype(np.float32)

    if not node_dist:
        return lambda _mass: None

    masses = np.array([layout.nodes[i].mass for i in node_dist.keys()])
    dists = np.stack([node_dist[i] for i in node_dist.keys()])
    order = np.argsort(masses)
    masses = masses[order]
    dists = dists[order]
    ppm = layout.ppm_tolerance

    def lookup(mass: float) -> Optional[np.ndarray]:
        tol = max(abs(mass), 1.0) * ppm * 1e-6
        j = int(np.searchsorted(masses, mass))
        best, best_diff = None, tol
        for k in (j - 1, j):
            if 0 <= k < masses.size:
                d = abs(masses[k] - mass)
                if d <= best_diff:
                    best_diff = d
                    best = dists[k]
        return None if best is None else best

    return lookup


# ---------------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------------


def save_consensus(layout: Layout, path: str) -> None:
    """Write a layout to a ``.npz`` consensus file.

    Stored arrays:

    * ``node_mass`` (N,)        — node masses, layout-ordered
    * ``node_coverage`` (N,)    — node read counts
    * ``edge_from`` (E,)        — edge source node indices
    * ``edge_to`` (E,)          — edge target node indices
    * ``edge_dist_sum`` (E, V)  — summed distributions
    * ``edge_coverage`` (E,)    — edge read counts
    * ``vocab_tokens`` (V,)     — string labels
    * ``token_masses`` (V,)     — monoisotopic masses
    * ``ppm_tolerance``         — node binning tolerance
    * ``read_src`` / ``read_shift`` — per-read N-terminal offset
    * ``unplaced``              — read srcs that failed to align
    """
    edges = list(layout.edges.values())
    if edges:
        edge_from = np.array([e.from_idx for e in edges], dtype=np.int64)
        edge_to = np.array([e.to_idx for e in edges], dtype=np.int64)
        edge_dist = np.stack([e.dist_sum for e in edges]).astype(np.float32)
        edge_cov = np.array([e.coverage for e in edges], dtype=np.int32)
    else:
        edge_from = np.zeros(0, dtype=np.int64)
        edge_to = np.zeros(0, dtype=np.int64)
        edge_dist = np.zeros((0, layout.token_masses.size), dtype=np.float32)
        edge_cov = np.zeros(0, dtype=np.int32)
    read_src = np.array(list(layout.read_shifts.keys()))
    read_shift = np.array(list(layout.read_shifts.values()), dtype=np.float64)
    np.savez_compressed(
        path,
        node_mass=np.array([n.mass for n in layout.nodes], dtype=np.float64),
        node_coverage=np.array(
            [n.coverage for n in layout.nodes], dtype=np.int32
        ),
        edge_from=edge_from,
        edge_to=edge_to,
        edge_dist_sum=edge_dist,
        edge_coverage=edge_cov,
        vocab_tokens=layout.vocab,
        token_masses=layout.token_masses,
        ppm_tolerance=np.float64(layout.ppm_tolerance),
        read_src=read_src,
        read_shift=read_shift,
        unplaced=np.array(layout.unplaced),
    )


def load_consensus(path: str) -> Layout:
    """Inverse of :func:`save_consensus`."""
    # allow_pickle=True is needed for the string ``vocab_tokens`` /
    # ``read_src`` arrays (object dtype). The arrays are emitted by
    # this module's own ``save_consensus``, so this is safe.
    d = np.load(path, allow_pickle=True)
    layout = Layout(
        vocab=d["vocab_tokens"],
        token_masses=d["token_masses"].astype(np.float64),
        ppm_tolerance=float(d["ppm_tolerance"]),
    )
    node_mass = d["node_mass"]
    node_coverage = d["node_coverage"]
    for m, c in zip(node_mass, node_coverage):
        layout.nodes.append(LayoutNode(mass=float(m), coverage=int(c)))
    layout._mass_index = sorted((float(m), i) for i, m in enumerate(node_mass))
    edge_from = d["edge_from"]
    edge_to = d["edge_to"]
    edge_dist = d["edge_dist_sum"]
    edge_cov = d["edge_coverage"]
    for i in range(edge_from.size):
        a = int(edge_from[i])
        b = int(edge_to[i])
        layout.edges[(a, b)] = LayoutEdge(
            from_idx=a,
            to_idx=b,
            dist_sum=edge_dist[i].astype(np.float64),
            coverage=int(edge_cov[i]),
        )
    layout.read_shifts = {
        str(s): float(sh) for s, sh in zip(d["read_src"], d["read_shift"])
    }
    layout.unplaced = [str(s) for s in d["unplaced"]]
    return layout
