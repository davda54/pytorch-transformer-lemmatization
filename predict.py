#!/usr/bin/env python3
import torch
import numpy as np

from morpho_dataset import MorphoDataset
from model import Model

VERSION = 'v3.0'


if __name__ == "__main__":
    import argparse
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--cle_layers", default=3, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=96, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path.")
    parser.add_argument("--dim", default=296, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
    parser.add_argument("--duz", default=0.1, type=float, help="Davsonův Ultimátní Zapomínák rate.")
    parser.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing of the cross-entropy loss.")
    parser.add_argument("--layers", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    parser.add_argument("--max_batch_size", default=60 * 1000, type=int, help="Max length of sentence in training.")
    parser.add_argument("--max_pos_len", default=8, type=int, help="Maximal length of the relative positional representation.")
    parser.add_argument("--warmup_steps", default=16000, type=int, help="Learning rate warmup.")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint", "evaluate_each", "max_batch_size", "skip_logging"]))
    args.directory = f"{args.base_directory}/models/{VERSION}_{architecture}"
    
    # Load the data
    morpho = MorphoDataset("czech_pdt", args.base_directory, add_bow_eow=True)

    # Create the network and train
    num_source_chars = len(morpho.train.data[morpho.train.FORMS].alphabet)
    num_target_chars = len(morpho.train.data[morpho.train.LEMMAS].alphabet)
    num_target_tags = len(morpho.train.data[morpho.train.TAGS].words)

    network = Model(args, num_source_chars, num_target_chars, num_target_tags).cuda()
    state = torch.load(f"{args.directory}/{args.checkpoint}")
    network.load_state_dict(state['state_dict'])

    network.eval()
    data = morpho.test

    batches_done = 0
    with torch.no_grad():
        lemma_sentences = []
        tag_sentences = []
        for b, (batch, batch_size) in enumerate(data.batches(args.max_batch_size)):
            lemmas, tags = network.predict_to_list(batch, data)
            lemma_sentences += lemmas
            tag_sentences += tags
            batches_done += batch_size
            
            print(f"\r{int(batches_done / data.size() * 100):d} %", end='', flush=True)

    out_path = "lemmatizer_test.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(lemma_sentences):
            for j in range(len(data.data[data.FORMS].word_strings[i])):
                lemma = []
                for c in map(int, sentence[j]):
                    if c == MorphoDataset.Factor.EOW: break
                    lemma.append(data.data[data.LEMMAS].alphabet[c])

                tag = data.data[data.TAGS].words[tag_sentences[i][j]]
                
                print(data.data[data.FORMS].word_strings[i][j], "".join(lemma), tag, sep="\t", file=out_file)
            print(file=out_file)

    print("\ndone")