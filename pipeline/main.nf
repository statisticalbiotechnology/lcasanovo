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

// Fetch Herceptin reference sequences from UniProt (heavy + light chain)
process FETCH_REFERENCE {
    publishDir "${params.outdir}/reference", mode: 'copy'

    output:
    path "reference.fasta"

    script:
    """
    python3 - <<'EOF'
import urllib.request

# Herceptin (trastuzumab): P0DOX7 = heavy chain, P0DOX8 = light chain
accessions = ['P0DOX7', 'P0DOX8']
with open('reference.fasta', 'w') as fh:
    for acc in accessions:
        url = f"https://www.uniprot.org/uniprot/{acc}.fasta"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            fh.write(resp.read().decode())
print("Fetched Herceptin heavy and light chain sequences from UniProt")
EOF
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
    }
}
