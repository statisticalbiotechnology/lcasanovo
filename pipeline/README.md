# lcasanovo Nextflow Pipeline

Downloads MS/MS spectra from PRIDE by PXD accession number and runs `casanovo sequence` on every spectrum file.

## Requirements

- [Nextflow](https://www.nextflow.io/) ≥ 23.04
- `wget` (for file downloads)
- `casanovo` reachable in `PATH` (or via container/conda profile)
- *(optional)* `msconvert` or the pwiz Docker image for `.raw` conversion

## Quick start

```bash
# From a PRIDE accession
nextflow run main.nf --pxd PXD012345 --outdir results/

# Multiple accessions
nextflow run main.nf --pxd "PXD012345,PXD067890" --outdir results/

# From a file listing accessions (one per line)
nextflow run main.nf --pxd accessions.txt --outdir results/

# From a local directory of spectrum files
nextflow run main.nf --input_dir /data/myexperiment/ --outdir results/

# Mix: local files and a PRIDE download together
nextflow run main.nf --pxd PXD012345 --input_dir /data/myexperiment/ --outdir results/

# With a custom model and config
nextflow run main.nf \
    --pxd PXD012345 \
    --model /path/to/weights.ckpt \
    --config /path/to/config.yaml \
    --outdir results/
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pxd` | — | PXD accession(s), comma-separated or a text file (one per line) |
| `--input_dir` | — | Directory of local spectrum files (mzML, mzXML, MGF, or Thermo raw) |
| `--model` | auto-download | Path or URL to casanovo model weights |
| `--config` | casanovo default | Path to casanovo YAML config |
| `--outdir` | `results` | Output directory |

At least one of `--pxd` or `--input_dir` is required. Both can be given together.

## Execution profiles

```bash
# Local, no containers (casanovo in PATH)
nextflow run main.nf -profile local --pxd PXD012345

# Local with Docker (GPU)
nextflow run main.nf -profile docker --pxd PXD012345

# SLURM cluster with Singularity
nextflow run main.nf -profile slurm --pxd PXD012345
```

## Output layout

```
results/
├── raw/
│   └── PXD012345/           # downloaded spectrum files (cached across runs)
├── predictions/
│   └── PXD012345/
│       ├── run1.mztab        # per-file predictions
│       └── run1.log
├── all_predictions.tsv       # merged PSM table across all files
└── pipeline_info/            # Nextflow timeline, report, trace, DAG
```

## Resume a failed run

```bash
nextflow run main.nf --pxd PXD012345 --outdir results/ -resume
```

The `storeDir` directive caches downloaded files, so only missing files are re-fetched on resume.
