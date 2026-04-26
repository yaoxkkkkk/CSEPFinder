# CSEPFinder

CSEPFinder is a Snakemake-based workflow for identifying candidate secreted effector proteins (CSEPs) from protein FASTA files or genome annotation files.

The workflow integrates four signal peptide / topology prediction tools:

- SignalP6
- PrediSi
- Phobius
- DeepTMHMM

Final CSEP candidates are selected based on combined evidence from multiple prediction tools.

---

## 1. Features

CSEPFinder supports two input modes:

1. **Protein mode**  
   Use a primary transcript protein FASTA file directly.

2. **Genome annotation mode**  
   If the primary transcript protein FASTA file is not available, provide:
   - genome FASTA file
   - genome annotation file

The workflow can:

- Extract primary transcript protein sequences
- Predict signal peptides using SignalP6, PrediSi, Phobius and DeepTMHMM
- Remove proteins with transmembrane domains when required
- Integrate prediction results from multiple tools
- Generate final CSEP candidate lists
- Summarize signal peptide cleavage-site predictions from different software

Gzipped input files are supported.

---

## 2. Dependencies

The following software is required:

- Java
- [seqkit](https://bioinf.shenwei.me/seqkit/)
- [SignalP6](https://services.healthtech.dtu.dk/services/SignalP-6.0/)
- [PrediSi](http://predisi.de/)
- [Phobius](https://phobius.sbc.su.se/)
- [DeepTMHMM](https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/)
- Snakemake
- Conda / Mamba

The workflow is designed to run with Snakemake using Conda environments and, optionally, Singularity.

---

## 3. DeepTMHMM patch

DeepTMHMM loads several model files relative to its execution directory. This can cause path errors when DeepTMHMM is executed inside a Snakemake workflow from a different working directory.

To avoid this issue, please replace the original DeepTMHMM scripts with the patched versions provided in:

```text
Replace_script/
````

Replace the following files in your DeepTMHMM installation directory:

```text
predict.py
utils.py
```

For example:

```bash
cd /path/to/DeepTMHMM/DeepTMHMM-Academic-License-v1.0

cp predict.py predict.py.bak
cp utils.py utils.py.bak

cp /path/to/CSEPFinder/Replace_script/predict.py ./
cp /path/to/CSEPFinder/Replace_script/utils.py ./
```

The patched version supports an explicit model directory argument:

```bash
--model-dir /path/to/DeepTMHMM/DeepTMHMM-Academic-License-v1.0
```

This makes DeepTMHMM compatible with Snakemake execution from arbitrary working directories.

---

## 4. Input files

### 4.1 Protein mode

If the primary transcript protein FASTA file is available, provide it directly in the configuration file.

Example:

```yaml
protein_file: "input/primary_transcripts.pep.fa"
```

### 4.2 Genome annotation mode

If the primary transcript protein FASTA file is not available, provide the genome FASTA file and annotation file.

Example:

```yaml
genome_file: "input/genome.fa"
annotation_file: "input/annotation.gff3"
```

Supported compressed formats:

```text
.fa
.fasta
.fa.gz
.fasta.gz
.gff
.gff3
.gff.gz
.gff3.gz
```

---

## 5. Configuration

Before running the workflow, edit:

```text
CSEPFinder_config.yaml
```

A typical configuration includes:

```yaml
ref: "your_reference_name"

protein_file: "input/primary_transcripts.pep.fa"

genome_file: ""
annotation_file: ""

deeptmhmm_folder: "/path/to/DeepTMHMM/DeepTMHMM-Academic-License-v1.0"

conda_env:
  signalP_env: "envs/signalP.yaml"
  DeepTMHMM_env: "envs/deeptmhmm.yaml"
```

If `protein_file` is provided, the workflow will use protein mode.

If `protein_file` is not provided, the workflow will use `genome_file` and `annotation_file` to generate the required protein FASTA file.

---

## 6. Usage

Run CSEPFinder with:

```bash
snakemake \
  --snakefile CSEPFinder.smk \
  --configfile CSEPFinder_config.yaml \
  -d ${your_working_dir} \
  --use-conda \
  --use-singularity \
  --nolock \
  --rerun-incomplete \
  --restart-time 3 \
  --rerun-triggers mtime
```

For a dry run:

```bash
snakemake \
  --snakefile CSEPFinder.smk \
  --configfile CSEPFinder_config.yaml \
  -d ${your_working_dir} \
  --use-conda \
  --use-singularity \
  --dry-run
```

---

## 7. Output files

The main output files are generated under:

```text
results/
```

Key outputs include:

Final CSEP candidate gene/protein IDs:

```text
results/{ref}.CSEP.txt
```

Final CSEP candidate protein sequences:

```text
results/{ref}.CSEP.fasta
```

Summary table of signal peptide cleavage-site or topology information predicted by different tools:

```text
results/{ref}.CSEP_signal_peptide_sites.tsv
```

Example format:

```text
gene id    signalP          Phobius    Predisi    DeepTMHMM
gene1      CS pos: 17-18	  c17/18o	   17         SSSSSSSSSOOOOOOO
```

Tool-specific outputs include:

```text
results/signalP/
results/Predisi/
results/Phobius/
results/DeepTMHMM/
```

---

## 8. Directory structure

A typical working directory after running the workflow:

```text
your_working_dir/
├── input/
├── results/
│   ├── signalP/
│   ├── Predisi/
│   ├── Phobius/
│   ├── DeepTMHMM/
│   ├── {ref}.CSEP.txt
│   ├── {ref}.CSEP.fasta
│   └── {ref}.CSEP_signal_peptide_sites.tsv
└── logs/
```

---

## 9. Notes

1. DeepTMHMM requires patched `predict.py` and `utils.py` files when used inside this Snakemake workflow.

2. SignalP6, PrediSi, Phobius and DeepTMHMM may require separate licenses or academic registration. Please install them according to their official instructions.

---

## 10. Citation

If you use CSEPFinder, please cite the original tools used in this workflow:

* SignalP6
* PrediSi
* Phobius
* DeepTMHMM
* seqkit
* Snakemake