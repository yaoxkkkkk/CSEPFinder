"""
This file is part of the DeepTMHMM project.

For license information, please see the README.txt file in the root directory.
"""

import numpy as np
import time
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from hashlib import md5
import os
from pathlib import Path
from tqdm import tqdm

import sys

from glob import glob

max_length_for_batching = 5000

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def chunk_with_constraints(l, n):
    chunks_list = chunks(l, n)
    seqs_to_remove = []
    chunks_to_use = []

    for chunk in chunks_list:
        for seq in chunk:
            if len(seq) > max_length_for_batching:
                chunks_to_use.append([seq])
                seqs_to_remove.append(seq)

        for seq in seqs_to_remove:
            if seq in chunk:
                chunk.remove(seq)

        if chunk:
            chunks_to_use.append(chunk)

    return chunks_to_use



prot_type_to_display_prot_type = {
    'TM': 'alpha TM',
    'SP+TM': 'alpha SP+TM',
    'SP': 'Globular + SP',
    'GLOB': 'Globular',
    'BETA': 'beta'
}

def type_id_to_string(type_id_list):
    types = []
    for type_id_np in type_id_list:
        type_id = int(type_id_np)
        if type_id == 0:
            types.append("TM")
        if type_id == 1:
            types.append("SP+TM")
        if type_id == 2:
            types.append("SP")
        if type_id == 3:
            types.append("GLOB")
        if type_id == 4:
            types.append("BETA")
    return types

def gff3(prot_id, seq_len, tmrs, region_strings, print_separator):
    outfile = ''
    outfile += "# " + prot_id + " Length: " + str(seq_len) + "\n"
    outfile += "# " + prot_id + " Number of predicted TMRs: " + str(tmrs) + "\n"

    for region in region_strings:
        outfile += "\t".join([prot_id] + [str(x) for x in region]) + "\t\t\t\t" + "\n"

    return outfile + '//\n' if print_separator else outfile

def load_model_from_disk(path, force_cpu=True):
    if force_cpu:
        # load model with map_location set to storage (main mem)
        model = torch.load(path, map_location=lambda storage, loc: storage)
        # flattern parameters in memory
        model.flatten_parameters()
        # update internal state accordingly
        model.use_gpu = False
    else:
        # load model using default map_location
        model = torch.load(path)
        model.flatten_parameters()
    return model

def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

def _resolve_model_dir(model_dir=None):
    """Resolve DeepTMHMM resource directory independent of current working directory."""
    script_dir = Path(__file__).resolve().parent
    resolved = Path(
        model_dir
        or os.environ.get("DEEPTMHMM_MODEL_DIR")
        or script_dir
    ).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DeepTMHMM model directory does not exist: {resolved}")
    return resolved


def _require_files(paths, label):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} file(s):\n" + "\n".join(missing)
        )


def generate_if1_embeddings_dir(npz_path, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings = np.load(npz_path)
    nr_of_seqs = len(embeddings.files)
    for idx, seq in enumerate(embeddings.files):
        data = embeddings[seq]
        embedding_path = f'{embeddings_dir}/{hash_aa_string(seq)}'
        if not os.path.exists(f'{embeddings_dir}/{hash_aa_string(seq)}'):
            with open(embedding_path, 'w') as emb_file:
                emb_file.write(data)
        print(f'Wrote seq {idx} of {nr_of_seqs}')


def generate_esm_embeddings(sequences, esm_embeddings_dir, repr_layers=33, chunk_size=3, model_dir=None):
    model_load_time = time.time()
    #model_data = torch.load('esm1b_model.pt', map_location='cpu')
    #esm_model, esm_alphabet = pretrained.load_model_and_alphabet_core(model_data)
    torch_load_time = time.time()
    print("Step 1/4 | Loading transformer model...")

    model_dir = _resolve_model_dir(model_dir)
    esm_args_path = model_dir / 'esm_model_args.pt'
    esm_alphabet_path = model_dir / 'esm_model_alphabet.pt'
    esm_state_dict_path = model_dir / 'esm_model_state_dict.pt'
    _require_files(
        [esm_args_path, esm_alphabet_path, esm_state_dict_path],
        label="DeepTMHMM ESM"
    )

    esm_args = torch.load(str(esm_args_path), map_location='cpu')
    esm_alphabet = torch.load(str(esm_alphabet_path), map_location='cpu')
    esm_model_state_dict = torch.load(str(esm_state_dict_path), map_location='cpu')
    #print(f'It took {time.time() - torch_load_time} to load torch vars')
    esm_model = ProteinBertModel(
        args=esm_args,
        alphabet=esm_alphabet
    )

    state_dict_time = time.time()
    esm_model.load_state_dict(esm_model_state_dict)
    #print(f'Loaded state dict in {time.time() - state_dict_time}s \n')

    #print(f'Loaded ESM model in {time.time() - model_load_time}s \n')

    os.makedirs(esm_embeddings_dir, exist_ok=True)

    print("\nStep 2/4 | Generating embeddings for sequences...")
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')
            esm_model = esm_model.cuda()
        esm_model.eval()

        batch_converter = esm_alphabet.get_batch_converter()
        for idx, seq in enumerate(tqdm(sequences, unit='seq', desc='Generating embeddings')):

            # if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}'):
            #     print("Already processed sequence")
            #     continue

            #print(f"Average sequence length in chunk: {sum( map(len, sequence_chunk) ) / len(sequence_chunk)} \n")

            embedding_seq_time = time.time()

            seqs = list([("seq", s) for s in [seq]])
            labels, strs, toks = batch_converter(seqs)
            repr_layers_list = [
                (i + esm_model.num_layers + 1) % (esm_model.num_layers + 1) for i in range(repr_layers)
            ]

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            minibatch_max_length = toks.size(1)

            tokens_list = []
            end = 0
            while end <= minibatch_max_length:
                start = end
                end = start + 1022
                if end <= minibatch_max_length:
                    # we are not on the last one, so make this shorter
                    end = end - 300
                tokens = esm_model(toks[:, start:end], repr_layers=repr_layers_list, return_contacts=False)[
                    "representations"][repr_layers - 1]
                tokens_list.append(tokens)

            out = torch.cat(tokens_list, dim=1)

            # set nan to zeros
            out[out != out] = 0.0

            res = out.transpose(0, 1)[1:-1]
            seq_embedding = res[:, 0]
            output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}', 'wb')
            torch.save(seq_embedding, output_file)
            output_file.close()
            #
            # for seq_idx, seq in enumerate(sequence_chunk):
            #     output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}', 'wb')
            #     #print(f"Saving file to: '{esm_embeddings_dir}/{hash_aa_string(seq)}'")
            #     seq_embedding = res[:, seq_idx]
            #     torch.save(seq_embedding, output_file)
            #     output_file.close()

            #print(f'Generated embedding for sequence chunk in {time.time() - embedding_seq_time}s')
            #print(f"Per sequence {(time.time() - embedding_seq_time) / chunk_size}s")

    # Manually delete the ESM model so it does not consume memory when running the DeepTMHMM models
    del esm_model

def write_probabilities_to_file(seq, seq_id, probs, output_dir):
    with open(f'{output_dir}/probabilities/{seq_id}_probs.csv', 'w') as probability_file:
        probability_file.write(f'# {seq_id}\n')
        probability_file.write(f'#AA,Beta,Periplasm,Membrane,Inside,Outside,Signal\n')
        for idx, res in enumerate(seq):
            line_to_write = f'{idx} {res}'
            for name in probs.keys():
                line_to_write += f',{round(float(probs[name][idx]), 5)}'
            probability_file.write(f'{line_to_write}\n')
