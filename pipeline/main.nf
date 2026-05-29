#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.pxd              = null   // comma-separated PXD accessions or text file
params.input_dir        = null   // directory of local spectrum files
params.model            = null   // lcasanovo model weights (.ckpt)
params.original_model   = null   // upstream casanovo model weights (.ckpt)
params.config           = null   // casanovo config YAML (optional)
params.outdir           = "results"
params.reference_fasta  = null   // reference protein FASTA for accuracy assessment
                                  // if null and --compare is set, fetches Herceptin from UniProt
params.compare          = false  // run both lcasanovo and upstream casanovo and compare
params.use_consensus    = false  // additionally run profile → assemble → redecode
params.ppm_tolerance    = 25.0   // consensus layout node ppm tolerance
params.redecode_alpha   = 0.5    // prior blend weight for second-pass decoding

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
SPECTRUM_EXTS = ['.raw', '.mzml', '.mzxml', '.mgf']

def parse_pxd_input(raw) {
    def f = file(raw)
    if (f.exists()) {
        return f.readLines().collect { it.trim() }.findAll { it && !it.startsWith('#') }
    }
    return raw.split(',').collect { it.trim() }
}

// ---------------------------------------------------------------------------
// Processes
// ---------------------------------------------------------------------------

// List spectrum files for an accession via PRIDE FTP
process FETCH_FILE_LIST {
    tag "$accession"

    input:
    val accession

    output:
    path "file_list_${accession}.tsv"

    script:
    """
    python3 - <<'EOF'
import ftplib, sys

SPECTRUM_EXTS = {'.raw', '.mzml', '.mzxml', '.mgf'}
FTP_BASE = '/pride/data/archive'
acc = "${accession}"

ftp = ftplib.FTP('ftp.pride.ebi.ac.uk')
ftp.login()

found_path = None
for year in ftp.nlst(FTP_BASE):
    try:
        months = ftp.nlst(year)
    except Exception:
        continue
    for month_path in months:
        candidate = f"{month_path}/{acc}"
        try:
            ftp.cwd(candidate)
            found_path = candidate
            break
        except Exception:
            pass
    if found_path:
        break

if not found_path:
    print(f"ERROR: {acc} not found on PRIDE FTP", flush=True)
    sys.exit(1)

files = ftp.nlst(found_path)
ftp.quit()

outfile = f"file_list_{acc}.tsv"
count = 0
with open(outfile, 'w') as fh:
    for fname in sorted(files):
        basename = fname.split('/')[-1]
        ext = '.' + basename.rsplit('.', 1)[-1].lower() if '.' in basename else ''
        if ext not in SPECTRUM_EXTS:
            continue
        url = f"ftp://ftp.pride.ebi.ac.uk{fname}"
        fh.write(f"{acc}\\t{basename}\\t{url}\\n")
        count += 1

print(f"{acc}: found {count} spectrum file(s)", flush=True)
if count == 0:
    sys.exit(1)
EOF
    """
}

// Download one spectrum file; cached across runs via storeDir
process DOWNLOAD_FILE {
    tag "$accession / $filename"
    storeDir "${params.outdir}/raw/${accession}"

    input:
    tuple val(accession), val(filename), val(url)

    output:
    tuple val(accession), path(filename)

    script:
    """
    wget -q --continue -O "${filename}" "${url}"
    """
}

// Convert a Thermo RAW file → mzML via ThermoRawFileParser biocontainer
process CONVERT_RAW {
    tag "$accession / ${raw_file.name}"
    container 'quay.io/biocontainers/thermorawfileparser:1.4.5--ha8f3691_0'

    input:
    tuple val(accession), path(raw_file)

    output:
    tuple val(accession), path("${raw_file.baseName}.mzML")

    script:
    """
    ThermoRawFileParser \
        --input="${raw_file}" \
        --output_directory=. \
        --format=1
    """
}

// Run lcasanovo (this fork) on one spectrum file
process CASANOVO_LCASANOVO {
    tag "$accession / ${spectrum_file.name}"
    publishDir "${params.outdir}/predictions/lcasanovo/${accession}", mode: 'copy'
    conda '/home/lukask/miniconda3/envs/lcasanovo'

    input:
    tuple val(accession), path(spectrum_file)

    output:
    tuple val(accession), val("${spectrum_file.baseName}"), path("${spectrum_file.baseName}.mztab"), emit: mztab
    tuple val(accession), path("${spectrum_file.baseName}.log"),                                     emit: log

    script:
    def model_arg  = params.model  ? "--model '${params.model}'"   : ""
    def config_arg = params.config ? "--config '${params.config}'" : ""
    """
    casanovo sequence \\
        ${model_arg} \\
        ${config_arg} \\
        --output_dir . \\
        --output_root "${spectrum_file.baseName}" \\
        "${spectrum_file}"
    """
}

// Run upstream casanovo (PyPI 5.x) on one spectrum file
process CASANOVO_ORIGINAL {
    tag "$accession / ${spectrum_file.name}"
    publishDir "${params.outdir}/predictions/original/${accession}", mode: 'copy'
    conda '/home/lukask/miniconda3/envs/casanovo_orig'

    input:
    tuple val(accession), path(spectrum_file)

    output:
    tuple val(accession), val("${spectrum_file.baseName}"), path("${spectrum_file.baseName}.mztab"), emit: mztab
    tuple val(accession), path("${spectrum_file.baseName}.log"),                                     emit: log

    script:
    def model_arg  = params.original_model ? "--model '${params.original_model}'" : ""
    def config_arg = params.config         ? "--config '${params.config}'"        : ""
    """
    casanovo sequence \\
        ${model_arg} \\
        ${config_arg} \\
        --output_dir . \\
        --output_root "${spectrum_file.baseName}" \\
        "${spectrum_file}"
    """
}

// --- Profile → Assemble → Re-decode (mass-coordinate consensus) ---
//
// Three-stage refinement of lcasanovo predictions:
//   1. PROFILE_LCA: emit per-spectrum softmax profiles to .npz.
//   2. ASSEMBLE_LCA: build a mass-coordinate consensus DAG across all
//      profiles for an accession; emit consensus.npz + bestpath.fasta.
//   3. REDECODE_LCA: re-run decoding biased by the consensus prior.
// Activated by `--use_consensus`; orthogonal to the upstream-vs-fork
// `--compare` baseline path.

// Per-spectrum first-pass profile (.npz)
process PROFILE_LCA {
    tag "$accession / ${spectrum_file.name}"
    publishDir "${params.outdir}/profiles/${accession}", mode: 'copy'
    conda '/home/lukask/miniconda3/envs/lcasanovo'

    input:
    tuple val(accession), path(spectrum_file)

    output:
    tuple val(accession), val("${spectrum_file.baseName}"), path("${spectrum_file.baseName}.npz"), emit: npz

    script:
    def model_arg  = params.model  ? "--model '${params.model}'"   : ""
    def config_arg = params.config ? "--config '${params.config}'" : ""
    """
    casanovo profile \\
        ${model_arg} \\
        ${config_arg} \\
        --output_dir . \\
        --output_root "${spectrum_file.baseName}" \\
        "${spectrum_file}"
    """
}

// Build the mass-coordinate consensus DAG across all profiles for an
// accession. Emits ``consensus.npz`` and the alignment-only candidate
// ``bestpath.fasta``. Pure data stage — no GPU required.
process ASSEMBLE_LCA {
    tag "$accession"
    publishDir "${params.outdir}/consensus/${accession}", mode: 'copy'
    conda '/home/lukask/miniconda3/envs/lcasanovo'

    input:
    tuple val(accession), path(npzs, stageAs: "profiles/*")

    output:
    tuple val(accession), path("${accession}.consensus.npz"), emit: consensus
    tuple val(accession), path("${accession}.bestpath.fasta"), emit: bestpath

    script:
    """
    casanovo assemble \\
        --output_dir . \\
        --output_root "${accession}" \\
        --ppm-tolerance ${params.ppm_tolerance} \\
        profiles/*.npz
    """
}

// Re-decode each spectrum file with the consensus DAG as prior.
process REDECODE_LCA {
    tag "$accession / ${spectrum_file.name}"
    publishDir "${params.outdir}/predictions/redecoded/${accession}", mode: 'copy'
    conda '/home/lukask/miniconda3/envs/lcasanovo'

    input:
    tuple val(accession), path(spectrum_file), path(consensus)

    output:
    tuple val(accession), val("${spectrum_file.baseName}"), path("${spectrum_file.baseName}.mztab"), emit: mztab
    tuple val(accession), path("${spectrum_file.baseName}.log"),                                     emit: log

    script:
    def model_arg  = params.model  ? "--model '${params.model}'"   : ""
    def config_arg = params.config ? "--config '${params.config}'" : ""
    """
    casanovo redecode \\
        ${model_arg} \\
        ${config_arg} \\
        --consensus "${consensus}" \\
        --alpha ${params.redecode_alpha} \\
        --output_dir . \\
        --output_root "${spectrum_file.baseName}" \\
        "${spectrum_file}"
    """
}

// Emit Herceptin (trastuzumab) reference sequences — published full-length
// heavy and light chains (DrugBank DB00072 / IMGT). Previously this fetched
// UniProt P0DOX7/P0DOX8, but those are generic kappa/lambda light chains
// (no heavy/Fc), which invalidated accuracy scoring against trastuzumab
// spectra. See prototypes/cross_check_reference.py for validation.
process FETCH_REFERENCE {
    publishDir "${params.outdir}/reference", mode: 'copy'

    output:
    path "reference.fasta"

    script:
    """
    cat > reference.fasta <<'EOF'
>trastuzumab_HC humanized IgG1 heavy chain (450 aa, DrugBank DB00072)
EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYA
DSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSSAS
TKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLY
SLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVF
LFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRV
VSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQV
SLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFS
CSVMHEALHNHYTQKSLSLSPGK
>trastuzumab_LC humanized kappa light chain (214 aa, DrugBank DB00072)
DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSR
FSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSD
EQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSK
ADYEKHKVYACEVTHQGLSSPVTKSFNRGEC
EOF
    echo "Wrote trastuzumab heavy (450 aa) + light (214 aa) reference"
    """
}

// Compare lcasanovo vs upstream casanovo predictions, scored against a reference FASTA
process COMPARE_PREDICTIONS {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path lcasanovo_mztabs,  stageAs: "lcasanovo/*"
    path original_mztabs,   stageAs: "original/*"
    path reference_fasta

    output:
    path "comparison.tsv"
    path "comparison_summary.tsv"

    script:
    """
    python3 - <<'EOF'
import glob, re
from difflib import SequenceMatcher

# ---- helpers ----------------------------------------------------------------

def read_mztab_psms(pattern):
    # returns {spectrum_id: sequence} from all mzTab files matching pattern
    psms = {}
    for path in glob.glob(pattern):
        fname = path.split('/')[-1].replace('.mztab', '')
        with open(path) as fh:
            for line in fh:
                if not line.startswith('PSM\\t'):
                    continue
                parts = line.rstrip('\\n').split('\\t')
                # mzTab PSM columns: PSM sequence PSM_ID ... spectra_ref
                seq = parts[1]
                psm_id = parts[2]
                uid = f"{fname}::{psm_id}"
                psms[uid] = seq
    return psms

def read_fasta(path):
    seqs = {}
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name:
        seqs[name] = ''.join(buf)
    return seqs

def best_aa_accuracy(predicted, reference_seqs):
    # slide predicted peptide over each reference protein; return best fraction match
    best = 0.0
    pred = predicted.upper().replace('I', 'L')  # I/L ambiguity
    plen = len(pred)
    for ref in reference_seqs.values():
        ref = ref.upper().replace('I', 'L')
        for start in range(len(ref) - plen + 1):
            window = ref[start:start + plen]
            matches = sum(a == b for a, b in zip(pred, window))
            score = matches / plen
            if score > best:
                best = score
    return best

# ---- load data --------------------------------------------------------------

lca  = read_mztab_psms("lcasanovo/*.mztab")
orig = read_mztab_psms("original/*.mztab")
refs = read_fasta("${reference_fasta}")

all_ids = sorted(set(lca) | set(orig))

rows = []
for uid in all_ids:
    lca_seq  = lca.get(uid,  '')
    orig_seq = orig.get(uid, '')
    lca_acc  = best_aa_accuracy(lca_seq,  refs) if lca_seq  else float('nan')
    orig_acc = best_aa_accuracy(orig_seq, refs) if orig_seq else float('nan')
    # sequence identity between the two tools for this spectrum
    if lca_seq and orig_seq:
        m = SequenceMatcher(None, lca_seq, orig_seq)
        identity = m.ratio()
    else:
        identity = float('nan')
    rows.append((uid, lca_seq, orig_seq, round(lca_acc, 4), round(orig_acc, 4), round(identity, 4)))

with open("comparison.tsv", 'w') as fh:
    fh.write("spectrum_id\\tlcasanovo_seq\\toriginal_seq\\tlcasanovo_aa_acc\\toriginal_aa_acc\\tseq_identity\\n")
    for r in rows:
        fh.write("\\t".join(str(x) for x in r) + "\\n")

# summary stats
import math
def mean(vals):
    v = [x for x in vals if not math.isnan(x)]
    return round(sum(v) / len(v), 4) if v else float('nan')

lca_accs  = [r[3] for r in rows]
orig_accs = [r[4] for r in rows]
identities = [r[5] for r in rows]

with open("comparison_summary.tsv", 'w') as fh:
    fh.write("metric\\tlcasanovo\\toriginal_casanovo\\n")
    fh.write(f"n_spectra\\t{sum(1 for r in rows if r[1])}\\t{sum(1 for r in rows if r[2])}\\n")
    fh.write(f"mean_aa_accuracy\\t{mean(lca_accs)}\\t{mean(orig_accs)}\\n")
    fh.write(f"pct_above_0.9_aa_acc\\t{round(sum(x>=0.9 for x in lca_accs if not math.isnan(x))/max(1,len(lca_accs))*100,1)}\\t{round(sum(x>=0.9 for x in orig_accs if not math.isnan(x))/max(1,len(orig_accs))*100,1)}\\n")
    fh.write(f"mean_seq_identity_between_tools\\t{mean(identities)}\\t-\\n")

print(f"Compared {len(rows)} spectra. See comparison.tsv and comparison_summary.tsv")
EOF
    """
}

// 4-way comparison: first-pass lcasanovo + upstream + redecoded + bestpath.
// Mirrors COMPARE_PREDICTIONS but adds the consensus-aware candidates.
process COMPARE_PREDICTIONS_FULL {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path lcasanovo_mztabs, stageAs: "lcasanovo/*"
    path original_mztabs,  stageAs: "original/*"
    path redecoded_mztabs, stageAs: "redecoded/*"
    path bestpath_fastas,  stageAs: "bestpath/*"
    path reference_fasta

    output:
    path "comparison_full.tsv"
    path "comparison_full_summary.tsv"
    path "bestpath_vs_reference.tsv"

    script:
    """
    python3 - <<'EOF'
import glob, math
from difflib import SequenceMatcher

def read_mztab_psms(pattern):
    psms = {}
    for path in glob.glob(pattern):
        fname = path.split('/')[-1].replace('.mztab', '')
        with open(path) as fh:
            for line in fh:
                if not line.startswith('PSM\\t'):
                    continue
                parts = line.rstrip('\\n').split('\\t')
                uid = f"{fname}::{parts[2]}"
                psms[uid] = parts[1]
    return psms

def read_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name:
        seqs[name] = ''.join(buf)
    return seqs

def best_aa_accuracy(predicted, reference_seqs):
    if not predicted:
        return float('nan')
    pred = predicted.upper().replace('I', 'L')
    plen = len(pred)
    if plen == 0:
        return float('nan')
    best = 0.0
    for ref in reference_seqs.values():
        ref = ref.upper().replace('I', 'L')
        if len(ref) < plen:
            continue
        for s in range(len(ref) - plen + 1):
            m = sum(a == b for a, b in zip(pred, ref[s:s+plen]))
            best = max(best, m / plen)
    return best

def mean(vals):
    v = [x for x in vals if not math.isnan(x)]
    return round(sum(v) / len(v), 4) if v else float('nan')

# ---- load -----------------------------------------------------------------

lca  = read_mztab_psms("lcasanovo/*.mztab")
orig = read_mztab_psms("original/*.mztab")
rdc  = read_mztab_psms("redecoded/*.mztab")
refs = read_fasta("${reference_fasta}")
all_ids = sorted(set(lca) | set(orig) | set(rdc))

# ---- per-PSM rows ---------------------------------------------------------

rows = []
for uid in all_ids:
    lca_s, orig_s, rdc_s = lca.get(uid, ''), orig.get(uid, ''), rdc.get(uid, '')
    lca_a  = best_aa_accuracy(lca_s,  refs)
    orig_a = best_aa_accuracy(orig_s, refs)
    rdc_a  = best_aa_accuracy(rdc_s,  refs)
    # delta vs first-pass lcasanovo (the natural baseline)
    delta = (rdc_a - lca_a) if (not math.isnan(rdc_a) and not math.isnan(lca_a)) else float('nan')
    ident_lca_rdc = (
        SequenceMatcher(None, lca_s, rdc_s).ratio()
        if lca_s and rdc_s else float('nan')
    )
    rows.append((uid, lca_s, orig_s, rdc_s,
                 round(lca_a, 4), round(orig_a, 4), round(rdc_a, 4),
                 round(delta, 4), round(ident_lca_rdc, 4)))

with open("comparison_full.tsv", 'w') as fh:
    fh.write("spectrum_id\\tlca_seq\\torig_seq\\tredecoded_seq\\t"
             "lca_aa_acc\\torig_aa_acc\\tredecoded_aa_acc\\t"
             "delta_redecoded_minus_lca\\tident_lca_vs_redecoded\\n")
    for r in rows:
        fh.write("\\t".join(str(x) for x in r) + "\\n")

# ---- aggregate summary ----------------------------------------------------

lca_accs  = [r[4] for r in rows]
orig_accs = [r[5] for r in rows]
rdc_accs  = [r[6] for r in rows]
deltas    = [r[7] for r in rows]

with open("comparison_full_summary.tsv", 'w') as fh:
    fh.write("metric\\tlcasanovo\\toriginal_casanovo\\tredecoded\\n")
    fh.write(f"n_spectra\\t{sum(1 for r in rows if r[1])}\\t"
             f"{sum(1 for r in rows if r[2])}\\t"
             f"{sum(1 for r in rows if r[3])}\\n")
    fh.write(f"mean_aa_accuracy\\t{mean(lca_accs)}\\t{mean(orig_accs)}\\t"
             f"{mean(rdc_accs)}\\n")
    for thresh in (0.5, 0.9):
        def above(accs, t):
            v = [x for x in accs if not math.isnan(x)]
            return round(sum(x >= t for x in v) / max(1, len(v)) * 100, 1)
        fh.write(f"pct_above_{thresh}\\t{above(lca_accs, thresh)}\\t"
                 f"{above(orig_accs, thresh)}\\t{above(rdc_accs, thresh)}\\n")
    fh.write(f"mean_delta_redecoded_minus_lca\\t-\\t-\\t{mean(deltas)}\\n")

# ---- bestpath candidates --------------------------------------------------

# Read every bestpath FASTA and score the candidate sequence against ref.
# The bestpath sequences may be long (up to the full antibody); use a
# longest-common-substring proxy against any reference, since "slide
# window" is meaningless when the candidate is the same length as the
# reference.
def lcs_fraction(query, target):
    # fraction of query that's in target as the longest common substring
    if not query or not target:
        return 0.0
    m = SequenceMatcher(None, query, target).find_longest_match(0, len(query), 0, len(target))
    return m.size / len(query)

with open("bestpath_vs_reference.tsv", 'w') as fh:
    fh.write("accession\\tcandidate_len\\theader\\tbest_ref_match\\tlcs_fraction\\n")
    for path in sorted(glob.glob("bestpath/*.fasta")):
        acc = path.split('/')[-1].replace('.bestpath.fasta', '').replace('.fasta', '')
        with open(path) as h:
            lines = h.read().splitlines()
        if not lines:
            continue
        header = lines[0][1:] if lines[0].startswith('>') else acc
        seq = ''.join(l for l in lines[1:] if not l.startswith('>'))
        seq_il = seq.upper().replace('I', 'L')
        best_name, best_frac = '-', 0.0
        for name, ref in refs.items():
            ref_il = ref.upper().replace('I', 'L')
            f = lcs_fraction(seq_il, ref_il)
            if f > best_frac:
                best_frac, best_name = f, name
        fh.write(f"{acc}\\t{len(seq)}\\t{header}\\t{best_name}\\t{round(best_frac, 4)}\\n")

print(f"4-way comparison done over {len(rows)} spectra")
EOF
    """
}

// Concatenate all mzTab PSM rows into a single summary table
process MERGE_LCASANOVO {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path(mztab_files, stageAs: "mztab/*")

    output:
    path "lcasanovo_predictions.tsv"

    script:
    """
    python3 - <<'EOF'
import glob

header, rows = None, []
for path in sorted(glob.glob("mztab/*.mztab")):
    with open(path) as fh:
        for line in fh:
            if line.startswith('PSH') and header is None:
                header = line
            elif line.startswith('PSM'):
                rows.append(line)

with open("lcasanovo_predictions.tsv", 'w') as out:
    if header:
        out.write(header)
    for r in rows:
        out.write(r)

print(f"Merged {len(rows)} PSM rows from {len(glob.glob('mztab/*.mztab'))} files")
EOF
    """
}

process MERGE_ORIGINAL {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path(mztab_files, stageAs: "mztab/*")

    output:
    path "original_predictions.tsv"

    script:
    """
    python3 - <<'EOF'
import glob

header, rows = None, []
for path in sorted(glob.glob("mztab/*.mztab")):
    with open(path) as fh:
        for line in fh:
            if line.startswith('PSH') and header is None:
                header = line
            elif line.startswith('PSM'):
                rows.append(line)

with open("original_predictions.tsv", 'w') as out:
    if header:
        out.write(header)
    for r in rows:
        out.write(r)

print(f"Merged {len(rows)} PSM rows from {len(glob.glob('mztab/*.mztab'))} files")
EOF
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------
workflow {
    if (!params.pxd && !params.input_dir) {
        error "Provide at least one input: --pxd 'PXD012345' or --input_dir /path/to/spectra"
    }

    // --- Remote path: PXD → download ---
    if (params.pxd) {
        pxd_files = FETCH_FILE_LIST(Channel.from(parse_pxd_input(params.pxd)))
            .splitCsv(sep: '\t', header: false)
            .map { row -> tuple(row[0], row[1], row[2]) }
            | DOWNLOAD_FILE
    } else {
        pxd_files = Channel.empty()
    }

    // --- Local path: directory → (label, file) tuples ---
    if (params.input_dir) {
        local_label = file(params.input_dir).name
        local_files = Channel
            .fromPath("${params.input_dir}/**")
            .filter { f -> SPECTRUM_EXTS.contains('.' + f.name.tokenize('.').last().toLowerCase()) }
            .map { f -> tuple(local_label, f) }
    } else {
        local_files = Channel.empty()
    }

    // --- Merge sources, convert RAW files ---
    all_files      = pxd_files.mix(local_files)
    raw_files      = all_files.filter { label, f -> f.name.toLowerCase().endsWith('.raw') }
    other_files    = all_files.filter { label, f -> !f.name.toLowerCase().endsWith('.raw') }
    spectrum_files = other_files.mix(CONVERT_RAW(raw_files))

    // --- lcasanovo predictions ---
    CASANOVO_LCASANOVO(spectrum_files)

    lca_mztabs = CASANOVO_LCASANOVO.out.mztab.map { acc, base, f -> f }.collect()
    MERGE_LCASANOVO(lca_mztabs)

    // --- Optional: mass-coordinate consensus refinement ---
    // Stages: PROFILE (per spectrum) → ASSEMBLE (per accession) →
    // REDECODE (per spectrum, biased by consensus).
    if (params.use_consensus) {
        PROFILE_LCA(spectrum_files)

        // Group all profiles by accession before assembly so the layout
        // builds across every spectrum from one antibody/sample.
        per_acc_profiles = PROFILE_LCA.out.npz
            .map { acc, base, npz -> tuple(acc, npz) }
            .groupTuple()
        ASSEMBLE_LCA(per_acc_profiles)

        // Combine each spectrum with its accession's consensus before
        // re-decoding.
        redecode_in = spectrum_files
            .combine(ASSEMBLE_LCA.out.consensus, by: 0)
        REDECODE_LCA(redecode_in)
        rdc_mztabs = REDECODE_LCA.out.mztab.map { acc, base, f -> f }.collect()
        bestpath_fastas = ASSEMBLE_LCA.out.bestpath.map { acc, f -> f }.collect()
    } else {
        rdc_mztabs      = Channel.empty().collect()
        bestpath_fastas = Channel.empty().collect()
    }

    // --- Optional: upstream casanovo comparison ---
    if (params.compare) {

        CASANOVO_ORIGINAL(spectrum_files)

        orig_mztabs = CASANOVO_ORIGINAL.out.mztab.map { acc, base, f -> f }.collect()
        MERGE_ORIGINAL(orig_mztabs)

        // Resolve reference FASTA
        if (params.reference_fasta) {
            ref_fasta = Channel.value(file(params.reference_fasta))
        } else {
            ref_fasta = FETCH_REFERENCE()
        }

        COMPARE_PREDICTIONS(lca_mztabs, orig_mztabs, ref_fasta)

        // With consensus on too, add the 4-way comparison (first-pass
        // lca + upstream + redecoded + bestpath candidate).
        if (params.use_consensus) {
            COMPARE_PREDICTIONS_FULL(
                lca_mztabs, orig_mztabs, rdc_mztabs, bestpath_fastas, ref_fasta
            )
        }
    }
}
