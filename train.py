#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

from morpho_dataset import MorphoDataset
from torch_attention import Model


def log_train_progress(epoch, loss, accuracy, accuracy_tags, learning_rate, progress):
    print('\r' + 140 * ' ', end='')  # clear line
    d, r = progress // 5, progress % 5
    loading_bar = d*'█' + (('░' if r < 2 else '▒' if r < 4 else '▓') + max(0, 19 - d)*'░' if progress < 100 else '')
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % │ lr: {learning_rate:1.6f} ║ {loading_bar} {progress:2d} %', end='', flush=True)

def log_train(epoch, loss, accuracy, accuracy_tags, out_file):
    print('\r' + 140 * ' ', end='')  # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ ', end='', flush=True)
    print(f'epoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ ', end='', flush=True, file=out_file)

def log_skipped_dev(out_file):
    print(flush=True)
    print(flush=True, file=out_file)

def log_dev(accuracy, accuracy_tags, learning_rate, out_file):
    print(f'dev acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'dev acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)


def log_mistakes(mistakes, out_file):
    max_before_len = max(len(mistake[0]) for mistake in mistakes)
    max_after_len = max(len(mistake[4]) for mistake in mistakes)
    max_original_len = max(len(mistake[1]) for mistake in mistakes)
    max_lemma_len = max(len(mistake[2]) for mistake in mistakes)
    for mistake in mistakes:
        print(f'{mistake[0].rjust(max_before_len)}\t║\t{mistake[1].ljust(max_original_len)}\t║\t{mistake[2].ljust(max_lemma_len)}\t║\t{mistake[3].ljust(max_lemma_len)}\t║\t{mistake[4].ljust(max_after_len)}', file=out_file)


def smooth_loss(pred, gold, smoothing):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot)


class LRDecay:
    def __init__(self, optimizers, args, initial_step=0):
        self.optimizers = optimizers
        self.dim = args.dim
        self.base_learning_rate = args.learning_rate
        self.warmup = args.warmup_steps
        self.step = initial_step

    def __call__(self):
        self.step += 1
        if self.step < self.warmup:
            learning_rate = self.dim ** (-0.5) * self.step * self.warmup ** (-1.5) * self.base_learning_rate
        else:
            learning_rate = 0.5*self.dim ** (-0.5) * self.step ** (-0.5) * self.base_learning_rate

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        return learning_rate


def get_example(indices, alphabet):
    example = []
    for i in indices:
        if i == MorphoDataset.Factor.BOW: continue
        if i == MorphoDataset.Factor.EOW: break
        example.append(alphabet[i])
    return ''.join(example)


def get_mistakes(truth_mask, dataset, inputs, predictions, targets):
    inputs, predictions, targets = inputs.numpy(), predictions.numpy(), targets.numpy()
    mistakes = []

    for i in range(inputs.shape[0]):
        if truth_mask[i].item() == 1: continue
        before = ""
        for k in range(max(0, i - 4), i):
            before += get_example(inputs[k, :], dataset.data[dataset.FORMS].alphabet) + ' '

        form = get_example(inputs[i, :], dataset.data[dataset.FORMS].alphabet)
        gold_lemma = get_example(targets[i, :], dataset.data[dataset.LEMMAS].alphabet)
        system_lemma = get_example(predictions[i, :], dataset.data[dataset.LEMMAS].alphabet)

        after = ""
        for k in range(min(inputs.shape[0] - 1, i + 1), min(inputs.shape[0] - 1, i + 5)):
            after += get_example(inputs[k, :], dataset.data[dataset.FORMS].alphabet) + ' '

        mistakes.append([before, form, gold_lemma, system_lemma, after])

    return mistakes


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the outputs.")
    parser.add_argument("--evaluate_each", default=3, type=int, help="After how many epoch do we want to evaluate.")
    parser.add_argument("--dim", default=296, type=int, help="Dimension of hidden layers.")
    parser.add_argument("--heads", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--layers", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
    parser.add_argument("--duz", default=0.1, type=float, help="Davsonův Ultimátní Zapomínák rate.")
    parser.add_argument("--cle_layers", default=3, type=int, help="CLE embedding layers.")
    parser.add_argument("--cnn_filters", default=96, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnn_max_width", default=5, type=int, help="Maximum CNN filter width.")
    parser.add_argument("--max_batch_size", default=60*1000, type=int, help="Max length of sentence in training.")
    parser.add_argument("--max_pos_len", default=8, type=int, help="Maximal length of the relative positional representation.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing of the cross-entropy loss.")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Initial learning rate multiplier.")
    parser.add_argument("--warmup_steps", default=16000, type=int, help="Learning rate warmup.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path.")
    args = parser.parse_args()

    architecture = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["directory", "base_directory", "epochs", "batch_size", "clip_gradient", "checkpoint", "evaluate_each", "max_batch_size"]))
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
    num_tags = len(morpho.train.data[morpho.train.TAGS].words)

    network = Model(args, num_source_chars, num_target_chars, num_tags).cuda()

    sparse_optimizer = torch.optim.SparseAdam([param for name, param in network.named_parameters() if name.split('.')[1] == 'embedding'], betas=(0.9, 0.98))
    dense_optimizer = torch.optim.Adam([param for name, param in network.named_parameters() if name.split('.')[1] != 'embedding'], betas=(0.9, 0.98))

    if args.checkpoint is not None:
        state = torch.load(f"{args.directory}/{args.checkpoint}")
        sparse_optimizer.load_state_dict(state['sparse_optimizer'])
        dense_optimizer.load_state_dict(state['dense_optimizer'])
        network.load_state_dict(state['state_dict'])
        initial_step = state['step']
        initial_epoch = state['epoch'] + 1
    else:
        initial_epoch, initial_step = 0, 0

    lr_decay = LRDecay([sparse_optimizer, dense_optimizer], args, initial_step)
    np.random.seed(987)
    
    for epoch in count(initial_epoch):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:

            #
            # TRAIN EPOCH
            #

            network.train()
            data = morpho.train

            batches_done = 0
            running_loss, total_images, correct, correct_tags, total_words = 0.0, 0, 0.0, 0.0, 0

            for b, (batch, batch_size) in enumerate(data.batches(args.max_batch_size)):
                learning_rate = lr_decay()

                sparse_optimizer.zero_grad()
                dense_optimizer.zero_grad()

                pred_lemmas, pred_tags, target_lemmas, target_tags = network(batch, data)

                tag_loss = smooth_loss(pred_tags, target_tags, smoothing=args.label_smoothing)
                lemma_loss = smooth_loss(pred_lemmas, target_lemmas, smoothing=args.label_smoothing)

                loss = tag_loss + lemma_loss
                loss.backward()
                sparse_optimizer.step()
                dense_optimizer.step()

                with torch.no_grad():
                    pred_tags = torch.argmax(pred_tags.data, 1).cpu()
                    target_tags = target_tags.cpu()
                    truth_mask_tags = pred_tags == target_tags
                    correct_tags += truth_mask_tags.sum().item()
                    total_words += target_tags.size(0)

                    target_lemmas = target_lemmas.cpu()
                    pred_lemmas = torch.argmax(pred_lemmas.data, 1).cpu()
                    truth_mask = pred_lemmas == target_lemmas
                    correct += truth_mask.sum().item()
                    total_images += truth_mask.size(0)
                    running_loss += loss.item() * truth_mask.size(0)

                    batches_done += batch_size
                    if b % 1 == 0:
                        log_train_progress(epoch, running_loss / total_images, correct / total_images * 100, correct_tags / total_words * 100, learning_rate, int(batches_done / data.size() * 100))
                      
                      
                    #print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached(), pred_lemmas.size(0), pred_tags.size(0), sep='\t')
                    #torch.cuda.reset_max_memory_allocated()
                    #torch.cuda.reset_max_memory_cached()
                    
            log_train(epoch, running_loss / total_images, correct / total_images * 100, correct_tags / total_words * 100, log_file)


            #
            # EVALUATE EPOCH
            #

            if epoch % args.evaluate_each != args.evaluate_each - 1:
               log_skipped_dev(log_file)
               continue

            network.eval()
            data = morpho.dev
            total_lemmas, correct, correct_tags, total_words, mistakes = 0, 0.0, 0.0, 0, []
            with torch.no_grad():
                for b, (batch, _) in enumerate(data.batches(6*args.max_batch_size)):
                    pred_lemmas, pred_tags, sources, target_lemmas, target_tags, _ = network.predict(batch, data)
                    pred_lemmas, sources, target_lemmas, target_tags = pred_lemmas.cpu(), sources.cpu(), target_lemmas[:, 1:].cpu(), target_tags.cpu()

                    pred_tags = torch.argmax(pred_tags.data, 1).cpu()
                    truth_mask_tags = pred_tags == target_tags
                    correct_tags += truth_mask_tags.sum().item()
                    total_words += target_tags.size(0)

                    mask = (target_lemmas != 0).to(torch.long)
                    resized_predictions = torch.cat((pred_lemmas, torch.zeros_like(target_lemmas)), dim=1)[:, :target_lemmas.size(1)]
                    truth_mask = (resized_predictions * mask == target_lemmas * mask).all(dim=1)

                    total_lemmas += target_lemmas.size(0)
                    correct += truth_mask.sum().item()

                    if len(mistakes) < 2500:
                        mistakes += get_mistakes(truth_mask, data, sources, pred_lemmas, target_lemmas)

                tag_accuracy = correct_tags / total_words * 100
                lemma_accuracy = correct / total_lemmas * 100
                log_dev(lemma_accuracy, tag_accuracy, learning_rate, log_file)

                with open(f"{args.directory}/mistakes_{epoch:03d}.txt", "w", encoding="utf-8") as mistakes_file:
                    log_mistakes(mistakes, mistakes_file)

                state = {
                    'epoch': epoch,
                    'step': lr_decay.step,
                    'state_dict': network.state_dict(),
                    'sparse_optimizer': sparse_optimizer.state_dict(),
                    'dense_optimizer': dense_optimizer.state_dict()
                }
                torch.save(state, f'{args.directory}/checkpoint_acc-{lemma_accuracy:2.3f}')