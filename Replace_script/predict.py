"""
This file is part of the DeepTMHMM project.

For license information, please see the README.txt file in the root directory.
"""

import numpy as np
import io
import time
import argparse
from Bio import SeqIO
import torch
import sys
from collections import OrderedDict
import os
from pathlib import Path
import shutil
from tqdm import tqdm


from utils import type_id_to_string, gff3, chunk_with_constraints, generate_esm_embeddings, load_model_from_disk, prot_type_to_display_prot_type, \
    write_probabilities_to_file
from experiments.tmhmm3.tm_util import original_labels_to_fasta, is_topologies_equal, crf_states, max_alpha_membrane_length, \
                                       max_beta_membrane_length, max_signal_length

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

# read inputs provided by user
# from experiments.tmhmm3.tm_util import decode

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count() + 2)
#print(f"CPUs: {os.cpu_count()}")
#print(f'Num threads: {torch.get_num_threads()}')
#print(f'Intraop num threads: {torch.get_num_interop_threads()}')

parser = argparse.ArgumentParser(description="DeepTMHMM v1.0")
parser.add_argument('--fasta', required=True, help="Path to a FASTA file containing the input sequences")
parser.add_argument('--output-dir', required=True, help='Path to write output files to')
parser.add_argument(
    '--model-dir',
    default=None,
    help=(
        "Directory containing DeepTMHMM/ESM model files. "
        "If not provided, use DEEPTMHMM_MODEL_DIR; "
        "if that is also unset, use the directory containing predict.py."
    )
)
parser.add_argument(
    '--force-overwrite',
    action='store_true',
    help=(
        "Remove output directory before running if it already exists. "
        "Default behavior allows an existing empty directory but refuses a non-empty directory."
    )
)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    args.model_dir
    or os.environ.get("DEEPTMHMM_MODEL_DIR")
    or SCRIPT_DIR
).resolve()

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"DeepTMHMM model directory does not exist: {MODEL_DIR}")

output_dir = Path(args.output_dir).resolve()

# Snakemake may create parent/output directories before the job starts.
# Allow an existing empty output directory, but protect non-empty directories
# unless --force-overwrite is explicitly requested.
if output_dir.exists():
    if args.force_overwrite:
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use --force-overwrite to remove it before running."
        )
else:
    output_dir.mkdir(parents=True, exist_ok=False)

probabilities_dir = output_dir / "probabilities"
probabilities_dir.mkdir(parents=True, exist_ok=True)

args.output_dir = str(output_dir)
args.fasta = str(Path(args.fasta).resolve())

input_str = open(args.fasta, "r").read()

if not input_str.strip().startswith(">") and input_str.strip() != "":
    input_str = "> Unnamed Protein \n" + input_str

input_str_buffer = io.StringIO(input_str)

protein_id_to_sequence = {}
protein_sequences = []
for idx, record in enumerate(SeqIO.parse(input_str_buffer, "fasta")):
    if record.seq == '':
        print(f"Error: No sequence for ID {record.id} found.")
        exit(1)
    protein_id_to_sequence[str(record.id)] = str(record.seq)
    protein_sequences.append(str(record.seq))

if len(protein_sequences) == 0:
    print("Error: No sequences found in FASTA input")
    exit(1)

protein_sequence_to_predicted_type = {}
protein_sequence_to_predicted_labels = {}
protein_sequence_to_predicted_topology = {}
protein_sequence_to_original_labels = {}

chunk_size = 1

types_count = {"TM": 0, "SP+TM": 0, "SP": 0, "GLOB": 0, "BETA": 0}

input_sequences = list(reversed(sorted(protein_sequences, key=len)))

print(f'Running DeepTMHMM on {len(input_sequences)} {"sequence" if len(input_sequences) == 1 else "sequences"}...')

#
embeddings_dir = f'{args.output_dir}/embeddings'
# First, generate embeddings.

generate_embedding_time = time.time()
generate_esm_embeddings(
    sequences=input_sequences,
    esm_embeddings_dir=embeddings_dir,
    chunk_size=1,
    model_dir=str(MODEL_DIR)
)

#print(f'Generated all embeddings in {time.time() - generate_embedding_time}s \n')

model_paths = [
    MODEL_DIR / 'deeptmhmm_cv_0.model',
    MODEL_DIR / 'deeptmhmm_cv_1.model',
    MODEL_DIR / 'deeptmhmm_cv_2.model',
    MODEL_DIR / 'deeptmhmm_cv_3.model',
    MODEL_DIR / 'deeptmhmm_cv_4.model',
]

missing_models = [str(p) for p in model_paths if not p.exists()]
if missing_models:
    raise FileNotFoundError(
        "Missing DeepTMHMM model file(s):\n" + "\n".join(missing_models)
    )

deeptmhmm_models_load_time = time.time()
models = []
for model_path in model_paths:
    models.append(load_model_from_disk(str(model_path)))
#print(f'It took {time.time() - deeptmhmm_models_load_time}s to load all 5 DeepTMHMM models \n')

for model in models:
    if torch.cuda.is_available():
        model.cuda()
        model.use_gpu = True
    else:
        model.use_gpu = False

    model.esm_embeddings_dir = embeddings_dir
    model.eval()

protein_sequence_chunks = chunk_with_constraints(input_sequences, chunk_size)

print(f"\nStep 3/4 | Predicting topologies for sequences in batches of {chunk_size}...")
full_prediction_loop_time = time.time()
with tqdm(total=len(input_sequences), desc='Topology prediction', unit='seq') as progress_bar:
    for sequence_chunk in protein_sequence_chunks:
        all_models_predicted_labels = {sequence: {} for sequence in sequence_chunk}
        all_models_predicted_types = {sequence: {} for sequence in sequence_chunk}
        all_models_predicted_topologies = {sequence: {} for sequence in sequence_chunk}
        all_models_crf_loss = {sequence: {} for sequence in sequence_chunk}

        average_sequence_length = sum( map(len, sequence_chunk) ) / len(sequence_chunk)
        #print("Average sequence length in chunk: ", average_sequence_length)
        #print('\n')

        with torch.no_grad():
            model_to_use_idx = None
            deeptmhmm_all_models_prediction_time = time.time()
            for model_idx, model in enumerate(models):
                single_model_prediction_time = time.time()
                predicted_labels, predicted_types, predicted_topologies, _, emissions, mask, predicted_crf_labels = model(sequence_chunk)
                padded_predicted_crf_labels = torch.nn.utils.rnn.pad_sequence(predicted_crf_labels)
                crf_loss_batch = model.crf_model(
                        emissions=emissions,
                        tags=padded_predicted_crf_labels,
                        mask=mask,
                        reduction='none'
                    )

                #print(f'It took {time.time() - single_model_prediction_time}s to generate prediction from a single model\n')

                for seq_idx, sequence in enumerate(sequence_chunk):
                    type_string = type_id_to_string([predicted_types[seq_idx]])[0]
                    labels = original_labels_to_fasta(predicted_labels[seq_idx])

                    all_models_predicted_types[sequence][model_idx] = type_string
                    all_models_predicted_labels[sequence][model_idx] = predicted_labels[seq_idx]
                    all_models_predicted_topologies[sequence][model_idx] = predicted_topologies[seq_idx]
                    all_models_crf_loss[sequence][model_idx] = crf_loss_batch[seq_idx]


            for sequence in sequence_chunk:
                topology_agreements = {model_idx: 0 for model_idx in range(len(models))}
                for model_idx in range(len(models)):
                    for other_model_idx in range(len(models)):
                        if model_idx == other_model_idx:
                            continue
                        model_topology = all_models_predicted_topologies[sequence][model_idx]
                        other_model_topology = all_models_predicted_topologies[sequence][other_model_idx]
                        if is_topologies_equal(model_topology, other_model_topology):
                            topology_agreements[model_idx] += 1

                max_topology_agreements = max(topology_agreements.values())
                tied_models_indexes = [idx for idx, value in topology_agreements.items() if value == max_topology_agreements]

                if len(tied_models_indexes) > 1:
                    # If tied, use the model with the highest log likelihood of the tied models.
                    model_to_use_idx = max(
                        {idx: loss for idx, loss in all_models_crf_loss[sequence].items() if idx in tied_models_indexes},
                        key=all_models_crf_loss[sequence].get
                    )

                else:
                    model_to_use_idx = tied_models_indexes[0]

                types_count[all_models_predicted_types[sequence][model_to_use_idx]] += 1

                protein_sequence_to_predicted_type[sequence] = all_models_predicted_types[sequence][model_to_use_idx]
                protein_sequence_to_predicted_labels[sequence] = original_labels_to_fasta(all_models_predicted_labels[sequence][model_to_use_idx])
                protein_sequence_to_original_labels[sequence] = all_models_predicted_labels[sequence][model_to_use_idx]
                protein_sequence_to_predicted_topology[sequence] = all_models_predicted_topologies[sequence][model_to_use_idx]

        progress_bar.update(len(sequence_chunk))
        #print(f'Ran all DeepTMHMM models in {time.time() - deeptmhmm_all_models_prediction_time}s\n')

#print(f'Ran full prediction loop in {time.time() - full_prediction_loop_time}s \n')
#print(f'It took {(time.time() - full_prediction_loop_time) / len(protein_sequences)}s per sequence \n')

print("\nStep 4/4 | Generating output...")

output_generation_time = time.time()

if len(protein_sequences) == 1:
    model = models[model_to_use_idx]
    model.use_marg_prob = True
    marg_probs, mask, _ = model.get_emissions_for_decoding(protein_sequences)

### Predicted Topologies
predicted_topologies = ''
for prot_id, seq in protein_id_to_sequence.items():
    predicted_topologies += f">{prot_id} | {protein_sequence_to_predicted_type[seq]}\n"
    predicted_topologies += f"{seq}\n"
    predicted_topologies += f"{protein_sequence_to_predicted_labels[seq]}\n"

with open(f"{args.output_dir}/predicted_topologies.3line", "w") as outfile:
    outfile.write(predicted_topologies)

markdown_outfile = open(f'{args.output_dir}/deeptmhmm_results.md' , 'w')

### TMRs
gff3_output = "##gff-version 3\n"

region_count = {"TMhelix": 0, "signal": 0, "inside": 0, "periplasm": 0, "outside": 0, 'Beta sheet': 0}

print_separator = True
for seq_idx, prot_seq_id in enumerate(protein_id_to_sequence.keys(), 1):

    sequence = protein_id_to_sequence[prot_seq_id]

    prot_seq_topology = protein_sequence_to_predicted_topology[sequence]
    prot_seq_labels = protein_sequence_to_predicted_labels[sequence]

    tmrs = 0
    for region in prot_seq_topology:
        if region[1] in (4, 5):
            tmrs += 1

    region_strings = []

    for idx, region in enumerate(prot_seq_topology):
        topology_category = region[1].item()
        # region[1]
        if idx == len(prot_seq_topology) - 1:
            end = len(prot_seq_labels)
        else:
            end = int(prot_seq_topology[idx+1][0])
        if topology_category == 0:
            region_str = "inside"
        elif topology_category == 1:
            region_str = "outside"
        elif topology_category == 2:
            region_str = "periplasm"
        elif topology_category == 3:
            region_str = "signal"
        elif topology_category == 4:
            region_str = "TMhelix"
        elif topology_category == 5:
            region_str = "Beta sheet"
        else:
            print("Error: unknown region", file=markdown_outfile)
        new_region = [region_str, str(int(region[0]) + 1), str(end)]
        region_strings.append(new_region)

        region_count[region_str] = region_count[region_str] + 1

    if seq_idx == len(protein_sequences):
        print_separator = False

    gff3_output += gff3(prot_seq_id, len(prot_seq_labels), tmrs, region_strings, print_separator)

with open(f"{args.output_dir}/TMRs.gff3", "w") as outfile:
    outfile.write(gff3_output)

print("## DeepTMHMM - Predictions", file=markdown_outfile)
#print("Predicted topologies (in .3lines and .gff format) can be downloaded by clicking \"Save Files\" above.")
print("Predicted topologies can be downloaded in [.gff3 format](TMRs.gff3) and [.3line format](predicted_topologies.3line)", file=markdown_outfile)

if not len(protein_sequences) == 1:
    #print("More than 1 protein submitted - omitting posterior probability graph.")
    pass
else:
    seq = protein_sequences[0]
    seq_id = list(protein_id_to_sequence.keys())[0]
    # We only have 1 protein so get the first element
    prot_type = list(protein_sequence_to_predicted_type.values())[0]
    marg_prob = marg_probs.cpu().exp().detach().numpy()[:, 0, :]

    marg_prob_plot = OrderedDict()
    marg_prob_plot["Beta"] = marg_prob[:, crf_states["Bop0"]:crf_states["Bop" + str(max_beta_membrane_length - 1)]+1].sum(axis=1) \
                             + marg_prob[:,crf_states["Bpo0"]:crf_states["Bpo" + str(max_beta_membrane_length - 1)]+1].sum(axis=1)

    marg_prob_plot["Periplasm"] = marg_prob[:, crf_states["Pb"]:crf_states["Pb"]+1].sum(axis=1)

    marg_prob_plot["Membrane"] = marg_prob[:, crf_states["Moi0"]:crf_states["Moi" +  str(max_alpha_membrane_length - 1)] + 1].sum(axis=1) \
                                 + marg_prob[:, crf_states["Mio0"]:crf_states["Mio" + str(max_alpha_membrane_length - 1)] + 1].sum(axis=1)

    marg_prob_plot["Inside"] = marg_prob[:, crf_states["I"]:crf_states["I"]+1].sum(axis=1)

    marg_prob_plot["Outside"] = marg_prob[:, crf_states["Om"]:crf_states["Om"]+1].sum(axis=1) + \
                                marg_prob[:, crf_states["Ob"]:crf_states["Ob"]+1].sum(axis=1)

    marg_prob_plot["Signal"] = marg_prob[:, crf_states["S0"]:crf_states["S" + str(max_signal_length - 1)] + 1].sum(axis=1)

    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    gs = gridspec.GridSpec(10, 1)

    plt.style.use('seaborn-whitegrid')

    # Most likely topology plot
    plt.subplot(gs[0:2, 0])

    lab_y = protein_sequence_to_original_labels[seq].reshape(-1).cpu()
    labels_for_signal_without_membrane = np.where(lab_y == 4, 0, lab_y)
    labels_for_signal = np.where(labels_for_signal_without_membrane == 3, 1, None)
    signal = np.ma.masked_not_equal(labels_for_signal, 1)

    membrane = np.where(lab_y == 4, 4, None)
    beta = np.where(lab_y == 5, 5, None)

    inside = np.where(lab_y == 0, 0, None)
    periplasm = np.where(lab_y == 2, 0, None)
    outside = np.where(lab_y == 1, 2, None)

    x_range = np.array(range(len(lab_y))) + 1


    plt.plot(x_range, outside, "#377eb8", linewidth=3)
    plt.plot(x_range, inside, "#f781bf", linewidth=3)
    plt.plot(x_range, periplasm, "#4daf4a", linewidth=3)
    plt.plot(x_range, signal, "#ff7f00", linewidth=3)


    if prot_type == 'BETA':
        plt.fill_between(x_range, 0, 2, where=beta == 5, color="#e41a1c", alpha=1)
        plt.yticks(np.arange(3), ('Periplasm', 'Signal/Beta', 'Outside'))

    elif prot_type == 'SP':
        plt.yticks(np.arange(3), ('Inside', 'Signal', 'Outside'))

    else:
        plt.fill_between(x_range, 0, 2, where=membrane == 4, color="#e41a1c", alpha=1)
        if 'SP' in prot_type:
            plt.yticks(np.arange(3), ('Inside', 'Signal/Membrane', 'Outside'))
        else:
            plt.yticks(np.arange(3), ('Inside', 'Membrane', 'Outside'))

    display_prot_type = prot_type_to_display_prot_type[prot_type]
    plt.title(f"DeepTMHMM - Most Likely Topology | Type: {display_prot_type}")

    # Posterior probability plot
    plt.subplot(gs[3:, 0])

    alpha_labels = ['Membrane', 'Inside', 'Outside', 'Signal']
    beta_labels = ['Beta', 'Periplasm', 'Outside', 'Signal']
    signal_labels = ['Signal', 'Outside']

    for idx, name in enumerate(marg_prob_plot.keys()):
        # if name == 'Signal' and ('SP' not in prot_type and prot_type != 'BETA'):
        #     continue
        #
        # if prot_type == 'SP':
        #     if name not in signal_labels:
        #         continue
        #
        # if prot_type == 'BETA':
        #     if name not in beta_labels:
        #         continue
        # else:
        #     if name not in alpha_labels:
        #         continue

        marg_time = time.time()
        label_probability_exceeds_threshold = False
        for res_idx, res in enumerate(marg_prob_plot[name]):
            if res > 0.01:
                label_probability_exceeds_threshold = True
                if res > 1.0:
                    marg_prob_plot[name][res_idx] = 1

        if not label_probability_exceeds_threshold:
            continue
        #print(f"Marg time {time.time() - marg_time}s")


        y = marg_prob_plot[name]
        x = np.array(range(len(y))) + 1

        color = ["#e41a1c", "#4daf4a", "#e41a1c", "#f781bf", "#377eb8", "#ff7f00"][idx]



        plt.plot(x, y, label=name, color=color)
        if name == "Membrane" or name == "Beta" :
            plt.fill_between(x, y, color=color)

    plt.legend(frameon=True, fontsize='small')
    plt.xlabel("Sequence")
    plt.ylabel("Probability")
    plt.title("DeepTMHMM - Posterior Probabilities")
    plt.subplots_adjust(hspace=3)
    axis = plt.gca()
    axis.set_ylim([0, 1.03])
    plt.savefig(f'{args.output_dir}/plot.png')

    print(f'![picture]({args.output_dir}/plot.png)', file=markdown_outfile)

    write_probabilities_to_file(seq, seq_id, marg_prob_plot, args.output_dir)
    print(f'You can download the probabilities used to generate this plot [here]({seq_id}_probs.csv)', file=markdown_outfile)

if len(protein_sequences) <= 50:
    topology_output = open(f"{args.output_dir}/predicted_topologies.3line", "r").read()
    print("### Predicted Topologies", file=markdown_outfile)
    print("```", file=markdown_outfile)
    print(topology_output, file=markdown_outfile)
    print("```", file=markdown_outfile)

    gff3_output = open(f"{args.output_dir}/TMRs.gff3", "r").read()
    print("\n", file=markdown_outfile)
    print("```", file=markdown_outfile)
    print(gff3_output.replace('//\n', ''), file=markdown_outfile)
    print("```", file=markdown_outfile)

if len(protein_sequences) > 1:
    print("### Job summary", file=markdown_outfile)
    print("```", file=markdown_outfile)
    print("Protein types:", file=markdown_outfile)
    for protein_type, protein_type_count in types_count.items():
        if protein_type_count:
            if len(protein_type) > 2:
                print(f"{protein_type}:\t\t" + str(protein_type_count), file=markdown_outfile)
            else:
                print(f"{protein_type}:\t\t\t" + str(protein_type_count), file=markdown_outfile)
    #
    # print("TM:\t\t\t" + str(types_count["TM"]))
    # print("SP:\t\t\t" + str(types_count["SP"]))
    # print("SP+TM:\t\t" + str(types_count["SP+TM"]))
    # print("GLOB:\t\t" + str(types_count["GLOB"]))
    # print("BETA:\t\t" + str(types_count["BETA"]))
    print('\n', file=markdown_outfile)
    print("Region types:", file=markdown_outfile)
    print("TMhelix:\t" + str(region_count["TMhelix"]), file=markdown_outfile)
    print("signal:\t\t" + str(region_count["signal"]), file=markdown_outfile)
    print("inside:\t\t" + str(region_count["inside"]), file=markdown_outfile)
    print("outside:\t" + str(region_count["outside"]), file=markdown_outfile)
    print("TMbeta:\t\t" + str(region_count["Beta sheet"]), file=markdown_outfile)
    print("```", file=markdown_outfile)

#print(f'It took {time.time() - output_generation_time}s to generate the output')

