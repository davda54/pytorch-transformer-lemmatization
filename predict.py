#!/usr/bin/env python3
import torch
import numpy as np

from morpho_dataset import MorphoDataset
from torch_attention import Model


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--evaluate_each", default=5, type=int, help="After how many epoch do we want to evaluate.")
    parser.add_argument("--batch_size", default=28, type=int, help="Batch size.")
    parser.add_argument("--dim", default=256, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--layers", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
    parser.add_argument("--duz", default=0.1, type=float, help="Davsonův Ultimátní Zapomínák rate.")
    parser.add_argument("--cle_layers", default=3, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=96, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--max_length", default=60, type=int, help="Max length of sentence in training.")
    parser.add_argument("--max_pos_len", default=8, type=int, help="Maximal length of the relative positional representation.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing of the cross-entropy loss.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    parser.add_argument("--warmup_steps", default=4000, type=int, help="Learning rate warmup.")
    parser.add_argument("--checkpoint", default="checkpoint_acc-98.677")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint"]))
    args.directory = f"{args.base_directory}/models/duz_attention_{architecture}"
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # Fix random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Load the data
    morpho = MorphoDataset("czech_pdt", args.base_directory, add_bow_eow=True)

    # Create the network and train
    num_source_chars = len(morpho.train.data[morpho.train.FORMS].alphabet)
    num_target_chars = len(morpho.train.data[morpho.train.LEMMAS].alphabet)
    num_target_tags = len(morpho.train.data[morpho.train.TAGS].words)

    network = Model(args, num_source_chars, num_target_chars, num_target_tags).cuda()
    state = torch.load(f"{args.directory}/{args.checkpoint}")
    network.load_state_dict(state['state_dict'])

    
    #
    # EVALUATE
    #
    
#     network.eval()
#     data = morpho.dev
#     total_lemmas, correct, correct_tags, total_words = 0, 0.0, 0.0, 0
#     with torch.no_grad():
#         for b, batch in enumerate(data.batches(128, 1000)):
#             pred_lemmas, pred_tags, sources, target_lemmas, target_tags, _ = network.predict(batch, data)
#             pred_lemmas, sources, target_lemmas, target_tags = pred_lemmas.cpu(), sources.cpu(), target_lemmas[:, 1:].cpu(), target_tags.cpu()

#             pred_tags = torch.argmax(pred_tags.data, 1).cpu()
#             truth_mask_tags = pred_tags == target_tags
#             correct_tags += truth_mask_tags.sum().item()
#             total_words += target_tags.size(0)

#             mask = (target_lemmas != 0).to(torch.long)
#             resized_predictions = torch.cat((pred_lemmas, torch.zeros_like(target_lemmas)), dim=1)[:, :target_lemmas.size(1)]
#             truth_mask = (resized_predictions * mask == target_lemmas * mask).all(dim=1)

#             total_lemmas += target_lemmas.size(0)
#             correct += truth_mask.sum().item()

#         tag_accuracy = correct_tags / total_words * 100
#         lemma_accuracy = correct / total_lemmas * 100
#         print(tag_accuracy, lemma_accuracy)
        
    #
    # PREDICT
    #

    network.eval()
    data = morpho.test
    size = data.size()
    
    with torch.no_grad():
        lemma_sentences = []
        tag_sentences = []
        for b, batch in enumerate(data.batches(128, 1000)):
            lemmas, tags = network.predict_to_list(batch, data)
            lemma_sentences += lemmas
            tag_sentences += tags
            
            print(f"\r{b / (size / 128) * 100:3.2f} %", end='', flush=True)

    print("INFERENCED")

    out_path = "lemmatizer_competition_test.txt"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(lemma_sentences):
            for j in range(len(data.data[data.FORMS].word_strings[i])):
                lemma = []
                for c in map(int, sentence[j]):
                    if c == MorphoDataset.Factor.EOW: break
                    lemma.append(data.data[data.LEMMAS].alphabet[c])

                tag = data.data[data.TAGS].words[tag_sentences[i][j]]
                
                print(data.data[data.FORMS].word_strings[i][j],
                      "".join(lemma),
                      tag,
                      sep="\t", file=out_file)
            print(file=out_file)