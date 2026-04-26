import os
import subprocess

# Function to check if a file is gzipped
def is_gzipped(file):
    return file.endswith(".gz")

# Function to get the base name without extensions
def get_basename(file):
    if is_gzipped(file):
        file = os.path.splitext(file)[0]  # Remove .gz
    return os.path.splitext(os.path.basename(file))[0]

# Check if protein_file is provided
if "protein_file" in config and config["protein_file"]:
    ref_basename = get_basename(config["protein_file"])
    protein_file = config["protein_file"]
elif "genome_file" in config and config["genome_file"] and "annotation_file" in config and config["annotation_file"]:
    ref_basename = get_basename(config["genome_file"])
    genome_file = config["genome_file"]
    annotation_file = config["annotation_file"]
else:
    raise ValueError("Either protein_file or both genome_file and annotation_file must be provided in the config file.")

rule all:
    input:
        f"results/{ref_basename}.CSEP.fasta",
        f"results/{ref_basename}.CSEP_signal_peptide_sites.tsv"

# If protein_file is provided, use the following rules
if "protein_file" in config and config["protein_file"]:
    rule file_treatment:
        input:
            protein_file=config["protein_file"]
        output:
            temp(f"{ref_basename}_treated.fasta")
        threads: 1
        resources:
            cpus_per_task=1
        shell:
            r"""
            seqkit replace \
            -p "\s.+" \
            {input.protein_file} \
            | sed 's/\*$//' \
            1> {output} \
            2> /dev/null
            """

    rule length_filter:
        input:
            protein_file=f"{ref_basename}_treated.fasta"
        output:
            f"{ref_basename}_shortpep.fasta"
        params:
            max_length=config["max_length"]
        threads: 1
        resources:
            cpus_per_task=1
        log:
            "logs/length_filter.log"
        shell:
            """
            seqkit seq \
            -M {params.max_length} \
            {input.protein_file} \
            1> {output} \
            2> {log}
            """

# If genome_file and annotation_file are provided, use the following rules
elif "genome_file" in config and config["genome_file"] and "annotation_file" in config and config["annotation_file"]:
    rule decompress_files:
        input:
            genome_file=config["genome_file"],
            annotation_file=config["annotation_file"]
        output:
            genome_file=temp(f"{ref_basename}.genome"),
            annotation_file=temp(f"{ref_basename}.annotation")
        threads: 2
        resources:
            cpus_per_task=2
        run:
            if is_gzipped(input.genome_file):
                shell("gunzip -c {input.genome_file} > {output.genome_file}")
            else:
                shell("cp {input.genome_file} {output.genome_file}")
            if is_gzipped(input.annotation_file):
                shell("gunzip -c {input.annotation_file} > {output.annotation_file}")
            else:
                shell("cp {input.annotation_file} {output.annotation_file}")

    rule primary_transcripts_extract:
        input:
            annotation_file=f"{ref_basename}.annotation"
        output:
            primary_only_file=temp(f"{ref_basename}.primaryTranscriptOnly.gff3")
        container:
            config["singularity"]["AGAT_image"]
        threads: 2
        resources:
            cpus_per_task=2
        log:
            "logs/primary_transcripts_extract.log"
        shell:
            """
            agat_sp_keep_longest_isoform.pl \
            -gff {input.annotation_file} \
            -o {output} \
            2> {log}
            """

    rule primary_sequence_extract:
        input:
            genome_file=f"{ref_basename}.genome",
            annotation_file=f"{ref_basename}.primaryTranscriptOnly.gff3"
        output:
            protein_file=f"{ref_basename}.protein_primaryTranscriptOnly.fasta"
        container:
            config["singularity"]["AGAT_image"]
        threads: 2
        resources:
            cpus_per_task=2
        log:
            "logs/primary_sequence_extract.log"
        shell:
            """
            agat_sp_extract_sequences.pl \
            -g {input.annotation_file} \
            -f {input.genome_file} \
            -t cds \
            -p \
            --cfs \
            --cis \
            -o {output} \
            2> {log}
            """

    rule length_filter:
        input:
            protein_file=f"{ref_basename}.protein_primaryTranscriptOnly.fasta"
        output:
            f"{ref_basename}_shortpep.fasta"
        params:
            max_length=config["max_length"]
        threads: 1
        resources:
            cpus_per_task=1
        log:
            "logs/length_filter.log"
        shell:
            """
            seqkit seq \
            -M {params.max_length} \
            {input.protein_file} \
            1> {output} \
            2> {log}
            """

# Common rules for both cases

rule Prediction_DeepTMHMM:
    input:
        protein_file=f"{ref_basename}_shortpep.fasta"
    output:
        dth_output="results/DeepTMHMM/wd/predicted_topologies.3line"
    params:
        output_dir="results/DeepTMHMM/wd/",
        dth_dir=config["deeptmhmm_folder"],
        model_dir=config["deeptmhmm_folder"]
    conda:
        config["conda_env"]["DeepTMHMM_env"]
    threads: 16
    resources:
        cpus_per_task=16,
        mem_mb=128000
    log:
        "logs/Prediction_DeepTMHMM.log"
    shell:
        r"""
        python "{params.dth_dir}/predict.py" \
            --fasta "{input.protein_file}" \
            --output-dir "{params.output_dir}" \
            --model-dir "{params.model_dir}" \
            --force-overwrite \
            > "{log}" 2>&1
        """

rule DeepTMHMM_res_parse:
    input:
        "results/DeepTMHMM/wd/predicted_topologies.3line"
    output:
        "results/DeepTMHMM/DeepTMHMM_summary.txt"
    conda:
        config["conda_env"]["DeepTMHMM_env"]
    threads: 1
    resources:
        cpus_per_task=1
    shell:
        r"""
        python - <<'PY'
import csv
import re

input_file = r"{input[0]}"
output_file = r"{output[0]}"

def find_tm_helices(topology):
    helices = []
    for match in re.finditer(r"M+", topology):
        start = match.start() + 1
        end = match.end()
        helices.append((start, end))
    return helices

with open(input_file) as f:
    lines = [line.rstrip("\n") for line in f]

if len(lines) % 3 != 0:
    raise ValueError(
        f"Input file {{input_file}} is not in valid 3-line format: "
        f"total lines = {{len(lines)}}, not divisible by 3."
    )

with open(output_file, "w", newline="") as out:
    writer = csv.writer(out, delimiter="\t")
    writer.writerow([
        "Sequence_ID",
        "Length",
        "Num_TM_helices",
        "TM_helices(start-end)",
        "Prediction",
        "Topology"
    ])

    for i in range(0, len(lines), 3):
        header = lines[i].strip()
        if not header.startswith(">"):
            raise ValueError(
                f"Expected header line starting with '>' at line {{i+1}}, got: {{header}}"
            )

        if " | " in header:
            seq_id, prediction = header[1:].split(" | ", 1)
        else:
            seq_id = header[1:]
            prediction = ""

        seq = lines[i + 1].strip()
        topology = lines[i + 2].strip()

        tm_coords = find_tm_helices(topology)
        tm_count = len(tm_coords)
        tm_str = ";".join([f"{{s}}-{{e}}" for s, e in tm_coords])

        writer.writerow([
            seq_id,
            len(seq),
            tm_count,
            tm_str,
            prediction,
            topology
        ])

print(f"Saved summary to {{output_file}}")
PY
        """

rule Extract_noTM_pep:
    input:
        protein_file=f"{ref_basename}_shortpep.fasta",
        dth_res="results/DeepTMHMM/DeepTMHMM_summary.txt"
    output:
        TM_pep_list="results/DeepTMHMM/protein_noTM.list",
        NoTM_pep=f"{ref_basename}_shortpep_noTM.fasta"
    threads: 1
    resources:
        cpus_per_task=1
    shell:
        r"""
        awk -F '\t' 'NR>1 && ($3=="0") {{print $1}}' {input.dth_res} > {output.TM_pep_list}

        seqkit grep -f {output.TM_pep_list} {input.protein_file} > {output.NoTM_pep}
        """

rule Prediction_signalP:
    input:
        protein_file=f"{ref_basename}_shortpep_noTM.fasta"
    output:
        "results/signalP/output.gff3"
    params:
        "results/signalP/"
    conda:
        config["conda_env"]["signalP_env"]
    threads: 16
    resources:
        cpus_per_task=16,
        mem_mb=128000
    log:
        "logs/Prediction_signalP.log"
    shell:
        """
        signalp6 \
        --fastafile {input} \
        --output_dir {params} \
        --format txt \
        --organism eukarya \
        --mode fast \
        2> {log}
        """

rule Prediction_Predisi:
    input:
        protein_file=f"{ref_basename}_shortpep_noTM.fasta"
    output:
        f"results/Predisi/{ref_basename}_Predisi.txt"
    params:
        predisi_folder=config["predisi_folder"]
    conda:
        config["conda_env"]["signalP_env"]
    threads: 16
    resources:
        cpus_per_task=16,
        mem_mb=128000
    log:
        "logs/Prediction_Predisi.log"
    shell:
        """
        java -cp {params.predisi_folder} \
        JSPP {params.predisi_folder}/matrices/eukarya.smx {input} {output} \
        2> {log}
        """

rule Prediction_Phobius:
    input:
        protein_file=f"{ref_basename}_shortpep_noTM.fasta"
    output:
        f"results/Phobius/{ref_basename}_Phobius.txt"
    conda:
        config["conda_env"]["signalP_env"]
    threads: 16
    resources:
        cpus_per_task=16,
        mem_mb=128000
    log:
        "logs/Prediction_Phobius.log"
    shell:
        r"""
        phobius.pl \
        -short {input.protein_file} \
        | sed '1s/SEQENCE //' \
        | awk 'BEGIN {{OFS="\t"}} {{print $1, $2, $3, $4}}' \
        1> {output} \
        2> {log}
        """

rule Phytocytokine_candidate_extraction:
    input:
        Phobius_file=f"results/Phobius/{ref_basename}_Phobius.txt",
        signalP_file="results/signalP/output.gff3",
        Predisi_file=f"results/Predisi/{ref_basename}_Predisi.txt",
        DeepTMHMM_file="results/DeepTMHMM/DeepTMHMM_summary.txt"
    output:
        Phobius_candidate=f"results/Phobius/{ref_basename}_Phobius_candidate.txt",
        signalP_candidate=f"results/signalP/{ref_basename}_signalP_candidate.txt",
        Predisi_candidate=f"results/Predisi/{ref_basename}_Predisi_candidate.txt",
        DeepTMHMM_candidate=f"results/DeepTMHMM/{ref_basename}_DeepTMHMM_candidate.txt"
    threads: 1
    resources:
        cpus_per_task=1
    shell:
        r"""
        awk -F "\t" '$3 == "Y" {{print $1}}' {input.Phobius_file} > {output.Phobius_candidate}
        awk -F "\t" '$3 == "signal_peptide" {{print $1}}' {input.signalP_file} > {output.signalP_candidate}
        awk -F "\t" '$3 == "Y" {{print $1}}' {input.Predisi_file} > {output.Predisi_candidate}
        awk -F "\t" '$5 == "SP" {{print $1}}' {input.DeepTMHMM_file} > {output.DeepTMHMM_candidate}
        """

rule Extract_common_candidates:
    input:
        Phobius_candidate=f"results/Phobius/{ref_basename}_Phobius_candidate.txt",
        signalP_candidate=f"results/signalP/{ref_basename}_signalP_candidate.txt",
        Predisi_candidate=f"results/Predisi/{ref_basename}_Predisi_candidate.txt",
        DeepTMHMM_candidate=f"results/DeepTMHMM/{ref_basename}_DeepTMHMM_candidate.txt"
    output:
        common_candidates=f"results/{ref_basename}.CSEP.txt"
    shell:
        r"""
        awk '
        NF > 0 {{
            key = FILENAME "\t" $1
            if (!(key in seen)) {{
                seen[key] = 1
                count[$1]++
            }}
        }}
        END {{
            for (id in count)
                if (count[id] >= 2)
                    print id
        }}
        ' {input.Phobius_candidate} \
          {input.signalP_candidate} \
          {input.Predisi_candidate} \
          {input.DeepTMHMM_candidate} \
          | sort > {output.common_candidates}
        """

rule Signal_peptide_extraction:
    input:
        common_candidates=f"results/{ref_basename}.CSEP.txt",
        protein_file=f"{ref_basename}_shortpep.fasta"
    output:
        f"results/{ref_basename}.CSEP.fasta"
    shell:
        """
        seqkit grep -f {input.common_candidates} {input.protein_file} > {output}
        """

rule Summarize_CSEP_signal_peptide_sites:
    input:
        common_candidates=f"results/{ref_basename}.CSEP.txt",
        signalP_result="results/signalP/prediction_results.txt",
        phobius_result=f"results/Phobius/{ref_basename}_Phobius.txt",
        predisi_result=f"results/Predisi/{ref_basename}_Predisi.txt",
        deeptmhmm_summary="results/DeepTMHMM/DeepTMHMM_summary.txt"
    output:
        summary=f"results/{ref_basename}.CSEP_signal_peptide_sites.tsv"
    threads: 1
    resources:
        cpus_per_task=1,
        mem_mb=8000,
        disk_mb=1000
    log:
        f"logs/Summarize_CSEP_signal_peptide_sites.{ref_basename}.log"
    run:
        import csv
        import re
        from pathlib import Path

        missing_value = "NA"

        Path(str(output.summary)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(log)).parent.mkdir(parents=True, exist_ok=True)

        def add_value(d, gene_id, value):
            """Store unique values and preserve insertion order."""
            if value is None:
                return
            value = str(value).strip()
            if value == "":
                return
            if gene_id not in d:
                d[gene_id] = []
            if value not in d[gene_id]:
                d[gene_id].append(value)

        def collapse(d, gene_id):
            if gene_id not in d or len(d[gene_id]) == 0:
                return missing_value
            return ";".join(d[gene_id])

        # 1. Read final common CSEP candidate IDs.
        candidate_ids = []
        with open(input.common_candidates) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gene_id = line.split()[0]
                candidate_ids.append(gene_id)

        candidate_set = set(candidate_ids)

        signalp = {}
        phobius = {}
        predisi = {}
        deeptmhmm = {}

        # 2. Parse SignalP result.
        # Example:
        # gene_id  SP  0.006275  0.993680  CS pos: 26-27. Pr: 0.7038
        with open(input.signalP_result) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 1:
                    continue

                gene_id = fields[0]
                if gene_id not in candidate_set:
                    continue

                m = re.search(r"CS pos:\s*\d+-\d+", line)
                if m:
                    add_value(signalp, gene_id, m.group(0))

        # 3. Parse Phobius result.
        # Example:
        # gene_id  0  Y  n5-16c20/21o
        # Extract c20/21o.
        with open(input.phobius_result) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 1:
                    continue

                gene_id = fields[0]
                if gene_id not in candidate_set:
                    continue

                m = re.search(r"c\d+/\d+o", line)
                if m:
                    add_value(phobius, gene_id, m.group(0))

        # 4. Parse PrediSi result.
        # Example:
        # gene_id  19  Y  0.6582622074058935
        # Extract second column.
        with open(input.predisi_result) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 2:
                    continue

                gene_id = fields[0]
                if gene_id not in candidate_set:
                    continue

                add_value(predisi, gene_id, fields[1])

        # 5. Parse DeepTMHMM summary.
        # Expected TSV:
        # Sequence_ID  Length  Num_TM_helices  TM_helices(start-end)  Prediction  Topology
        # Also tolerates comma-delimited old summary.
        with open(input.deeptmhmm_summary) as f:
            first_line = f.readline()
            delimiter = "\t" if "\t" in first_line else ","

        with open(input.deeptmhmm_summary) as f:
            reader = csv.reader(f, delimiter=delimiter)

            for row in reader:
                if not row:
                    continue

                if row[0] == "Sequence_ID":
                    continue

                if len(row) < 2:
                    continue

                gene_id = row[0]
                if gene_id not in candidate_set:
                    continue

                topology = row[-1]
                add_value(deeptmhmm, gene_id, topology)

        # 6. Write final table.
        with open(output.summary, "w", newline="") as out:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow([
                "gene id",
                "signalP",
                "Phobius",
                "Predisi",
                "DeepTMHMM"
            ])

            for gene_id in candidate_ids:
                writer.writerow([
                    gene_id,
                    collapse(signalp, gene_id),
                    collapse(phobius, gene_id),
                    collapse(predisi, gene_id),
                    collapse(deeptmhmm, gene_id)
                ])

        with open(log[0], "w") as log_out:
            log_out.write(f"Input common candidates: {len(candidate_ids)}\n")
            log_out.write(f"SignalP parsed candidates: {len(signalp)}\n")
            log_out.write(f"Phobius parsed candidates: {len(phobius)}\n")
            log_out.write(f"PrediSi parsed candidates: {len(predisi)}\n")
            log_out.write(f"DeepTMHMM parsed candidates: {len(deeptmhmm)}\n")
            log_out.write(f"Output written to: {output.summary}\n")