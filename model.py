import math
import numpy as np
import torch
import torch.nn as nn
from morpho_dataset import MorphoDataset

from tsalib import dim_vars
B, T, D, H, E = dim_vars('Batch SeqLength Dim NumHeads PositionalEmbedding')
T_K, T_Q = dim_vars('SeqLengthKey SeqLengthQuery')
S, W, C = dim_vars('Sentences Words Chars')


def arrange_char_pos_embedding(len_k: int, len_q: int, max_len: int, embedding: (E,D//H)) -> (T_Q,T_K,D//H):
    k: (T_K) = torch.arange(len_k, device='cuda')
    q: (T_Q) = torch.arange(len_q, device='cuda')
    indices: (T_Q,T_K) = k.view(1, -1) - q.view(-1, 1)
    indices.clamp_(-max_len, max_len).add_(max_len)

    return embedding[indices, :]

def arrange_word_pos_embedding(len_k: int, lens_q: int, max_len: int, embedding: (E,D//H)) -> (B,T_K,D//H):
    k: (T_K) = torch.arange(len_k, device='cuda')
    q: (B) = torch.repeat_interleave(lens_q - lens_q.cumsum(dim=0), lens_q, dim=0) + torch.arange(lens_q.sum(), device='cuda')
    indices: (B,T_K) = k.view(1, -1) - q.view(-1, 1)
    indices.clamp_(-max_len, max_len).add_(max_len)

    return embedding[indices, :]


class PositionLinear(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        w = torch.empty(in_dim, out_dim)
        torch.nn.init.normal_(w, std=0.015)
        self.w: (in_dim, out_dim) = nn.Parameter(w)
        self.b: (out_dim,) = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.out_dim,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        return x.view(*size_out)


class AttentionSublayer(nn.Module):
    def __init__(self, dimension, heads, max_pos_len, attention_dropout):
        super(AttentionSublayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.max_pos_len = max_pos_len
        self.scale = math.sqrt(dimension / heads)

        self.pos_embedding = nn.Parameter(torch.randn(2*max_pos_len + 1, dimension // heads, device='cuda', requires_grad=True))
        self.input_transform_q = nn.Linear(dimension, dimension, bias=False)
        self.input_transform_k = nn.Linear(dimension, dimension, bias=False)
        self.input_transform_v = nn.Linear(dimension, dimension, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_transform = nn.Linear(dimension, dimension, bias=False)

    def _split_heads(self, x: (B,T,D)) -> (B,H,T,D//H):
        x: (B,T,H,D//H) = x.view(x.size(0), -1, self.heads, self.dimension // self.heads)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x_q: (B,T,D), x_k: (B,T,D), x_v: (B,T,D), mask: (B,1,1,T), sentence_lengths=None):
        Q: (B,H,T_Q,D//H) = self._split_heads(self.input_transform_q(x_q))
        K: (B,H,T_K,D//H) = self._split_heads(self.input_transform_k(x_k))
        V: (B,H,T_K,D//H) = self._split_heads(self.input_transform_v(x_v))

        batch_size, _, len_q, _ = Q.size()
        len_k = K.size(2)

        logits: (B,H,T_Q,T_K) = torch.matmul(Q, K.transpose(2, 3))

        if sentence_lengths is None:
            arranged_pos: (T_Q,T_K,D//H) = arrange_char_pos_embedding(len_k, len_q, self.max_pos_len, self.pos_embedding)
            Q_t: (T_Q,B*H,D//H) = Q.permute(2, 0, 1, 3).view(len_q, batch_size * self.heads, -1)
            pos_logits: (T_Q,B*H,T_K) = torch.matmul(Q_t, arranged_pos.transpose(1, 2))
            pos_logits: (B,H,T_Q,T_K) = pos_logits.view(len_q, batch_size, self.heads, -1).permute(1, 2, 0, 3)
        else:
            arranged_pos: (B,T_K,D//H) = arrange_word_pos_embedding(len_k, sentence_lengths, self.max_pos_len, self.pos_embedding)
            pos_logits: (H,B,T_Q,T_K) = torch.matmul(Q.transpose(0, 1), arranged_pos.transpose(1, 2))
            pos_logits: (B,H,T_Q,T_K) = pos_logits.transpose(0, 1)

        logits: (B,H,T_Q,T_K) = (logits + pos_logits) / self.scale
        if mask is not None: logits.masked_fill_(mask, -np.inf)
        indices: (B,H,T_Q,T_K) = nn.functional.softmax(logits, dim=-1)
        indices: (B,H,T_Q,T_K) = self.attention_dropout(indices)
        combined: (B,H,T_Q,D//H) = torch.matmul(indices, V)

        heads_concat: (B,T_Q,H,D//H) = combined.permute(0, 2, 1, 3).contiguous()
        heads_concat: (B,T_Q,D) = heads_concat.view(heads_concat.size(0), -1, self.dimension)

        return self.output_transform(heads_concat)


class DenseSublayer(nn.Module):
    def __init__(self, dimension, dropout):
        super(DenseSublayer, self).__init__()

        self.linear_1 = PositionLinear(dimension, 4*dimension)
        self.relu = nn.ReLU()
        self.linear_2 = PositionLinear(4*dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: (B,T,D)) -> (B,T,D):
        x: (B,T,4*D) = self.relu(self.linear_1(x))
        return self.dropout(self.linear_2(x))


class EncoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout, max_pos_len, attention_dropout):
        super(EncoderLayer, self).__init__()

        self.attention = AttentionSublayer(dimension, heads, max_pos_len, attention_dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(dimension)

        self.dense_sublayer = DenseSublayer(dimension, dropout)
        self.layer_norm_2 = nn.LayerNorm(dimension)

    def forward(self, x: (B,T,D), mask: (B,T,D,D)):
        attention: (B,T,D) = self.attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout_1(attention))

        return self.layer_norm_2(x + self.dense_sublayer(x))


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout, max_pos_len, attention_dropout):
        super(DecoderLayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.attention_1 = AttentionSublayer(dimension, heads, max_pos_len, attention_dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(dimension)

        self.attention_2 = AttentionSublayer(dimension, heads, max_pos_len, attention_dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(dimension)

        self.attention_3 = AttentionSublayer(dimension, heads, max_pos_len, attention_dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(dimension)

        self.dense_sublayer = DenseSublayer(dimension, dropout)
        self.layer_norm_4 = nn.LayerNorm(dimension)

    def forward(self, char_encoder_output, word_encoder_output, x, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths):
        attention = self.attention_1(x, x, x, look_ahead_mask)
        x = self.layer_norm_1(x + self.dropout_1(attention))

        attention = self.attention_2(x, char_encoder_output, char_encoder_output, char_padding_mask)
        x = self.layer_norm_2(x + self.dropout_2(attention))

        attention = self.attention_3(x, word_encoder_output, word_encoder_output, word_padding_mask, sentence_lengths)
        x = self.layer_norm_3(x + self.dropout_3(attention))

        return self.layer_norm_4(x + self.dense_sublayer(x))


class Encoder(nn.Module):
    def __init__(self, dimension, heads, layers, dropout, max_pos_len, attention_dropout):
        super(Encoder, self).__init__()

        self.scale = float(math.sqrt(dimension))
        self.dropout = nn.Dropout(dropout)

        self.encoding = nn.ModuleList([EncoderLayer(dimension, heads, dropout, max_pos_len, attention_dropout) for _ in range(layers)])

    def forward(self, x, mask):
        x = x * self.scale
        x = self.dropout(x)

        for encoding in self.encoding:
            x = encoding(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_chars, dimension, heads, layers, dropout, duz, max_pos_len):
        super(Decoder, self).__init__()

        self.scale = float(math.sqrt(dimension))

        self.embedding = nn.Embedding(num_chars, dimension, padding_idx=MorphoDataset.Factor.PAD, sparse=True)
        self.dropout = nn.Dropout(dropout)
        self.char_dropout = nn.Dropout2d(duz)
        self.word_dropout = nn.Dropout2d(duz)

        self.decoding = nn.ModuleList([DecoderLayer(dimension, heads, dropout, max_pos_len, duz) for _ in range(layers)])

        self.classifier = nn.Linear(dimension, num_chars)
        #self.classifier.weight = self.embedding.weight # tie weights
        #self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, char_encoder_output, word_encoder_output, x, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths):
        x = self.embedding(x) * self.scale
        x = self.dropout(x)

        char_encoder_output = self.char_dropout(char_encoder_output.unsqueeze(0)).squeeze_(0)
        word_encoder_output = self.word_dropout(word_encoder_output)

        for decoding in self.decoding:
            x = decoding(char_encoder_output, word_encoder_output, x, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths)

        return self.classifier(x)


class WordEmbedding(nn.Module):
    def __init__(self, args, num_source_chars):
        super().__init__()

        self.word_dropout = nn.Dropout2d(args.duz)

        self.dim = args.dim
        self.embedding = nn.Embedding(num_source_chars, args.dim, padding_idx=MorphoDataset.Factor.PAD, sparse=True)

        convolutions = []
        for width in range(3, args.cnn_max_width + 1, 2):
            conv = [
                nn.Conv1d(args.dim, args.cnn_filters, kernel_size=width, stride=1, padding=(width - 1) // 2),
                nn.ReLU()
            ]
            for _ in range(1, args.cle_layers):
                conv.append(nn.Conv1d(args.cnn_filters, args.cnn_filters, kernel_size=width, stride=1, padding=(width-1)//2))
                conv.append(nn.ReLU())
            convolutions.append(nn.Sequential(*conv))

        self.convolutions = nn.ModuleList(convolutions)
        self.conv_linear = nn.Linear((args.cnn_max_width - 1)//2*args.cnn_filters, args.dim)
        self.conv_relu = nn.ReLU()
        self.combined_linear = nn.Linear(args.dim + MorphoDataset.Dataset.EMBEDDING_SIZE, args.dim)
        self.combined_relu = nn.ReLU()

    def forward(self, chars: (S,W,C), word_embeddings: (S,W), char_mask):
        word_embeddings: (S,W,D) = self.word_dropout(word_embeddings)

        sentences, words, char_len = chars.size()
        char_embedding: (S,W,C,D) = self.embedding(chars)
        char_embedding_flat: (S*W,D,C) = char_embedding.view(-1, char_len, self.dim).transpose(-2, -1)

        to_concat = []
        for convolution in self.convolutions:
            convoluted = convolution(char_embedding_flat)
            to_concat.append(convoluted.max(dim=-1).values)
        concated = torch.cat(to_concat, dim=-1)
        convoluted_char_embedding: (S,W,D) = self.conv_relu(self.conv_linear(concated)).view(sentences, words, -1)

        full_embedding: (S,W,D) = self.combined_linear(torch.cat([word_embeddings, convoluted_char_embedding], dim=-1))
        full_embedding: (S,W,D) = self.combined_relu(full_embedding)

        return chars[char_mask,:], char_embedding[char_mask,:,:], full_embedding


class Model(nn.Module):
    def __init__(self, args, num_source_chars, num_target_chars, num_target_tags):
        super().__init__()

        self._input_embedding = WordEmbedding(args, num_source_chars)
        self._encoder = Encoder(args.dim, args.heads, args.layers // 2, args.dropout, args.max_pos_len, args.duz)
        self._encoder_sentence = Encoder(args.dim, args.heads, args.layers, args.dropout, args.max_pos_len, args.duz)
        self._decoder = Decoder(num_target_chars, args.dim, args.heads, args.layers, args.dropout, args.duz, args.max_pos_len)
        self._tag_classifier = nn.Linear(args.dim, num_target_tags)

    def _create_look_ahead_mask(self, target: (B,T)) -> (B,1,T,T):
        look_ahead: (T,T) = torch.ones(target.size(1), target.size(1), device='cuda', dtype=torch.uint8).triu_(1)
        padding: (B,1,1,T) = self._create_padding_mask(target)
        return torch.max(padding, look_ahead)

    def _create_padding_mask(self, seq: (B,T)) -> (B,1,1,T):
        seq = seq == 0
        seq.unsqueeze_(1).unsqueeze_(1)
        return seq

    def _gather_batch(self, batch, dataset):
        source_charseq_ids = torch.LongTensor(batch[dataset.FORMS].charseq_ids)
        source_charseqs = torch.LongTensor(batch[dataset.FORMS].charseqs)
        source_word_ids = torch.LongTensor(batch[dataset.FORMS].word_ids)
        source_words = torch.FloatTensor(batch[dataset.FORMS].word_embeddings)
        target_charseq_ids = torch.LongTensor(batch[dataset.LEMMAS].charseq_ids)
        target_charseqs = torch.LongTensor(batch[dataset.LEMMAS].charseqs)
        target_tags = torch.LongTensor(batch[dataset.TAGS].word_ids)
        source_mask = source_charseq_ids != 0

        sources = source_charseqs[source_charseq_ids, :]
        targets = target_charseqs[target_charseq_ids, :][target_charseq_ids != 0]
        encoder_mask = self._create_padding_mask(sources[source_mask])

        sentence_lenghts = (source_word_ids != 0).sum(dim=1)
        encoder_sentence_mask = self._create_padding_mask(source_word_ids)

        return sources.cuda(), source_mask.cuda(), targets.cuda(), source_words.cuda(), encoder_mask.cuda(), encoder_sentence_mask.cuda(), sentence_lenghts.cuda(), target_tags.cuda()

    def forward(self, batch, dataset):
        sources, source_mask, targets, sentences, char_encoder_mask, word_encoder_mask, sentence_lengths, target_tags = self._gather_batch(batch, dataset)
        targets_in, targets_out = targets[:, :-1], targets[:, 1:]

        decoder_combined_mask = self._create_look_ahead_mask(targets_in)

        _, embedded_chars, embedded_words = self._input_embedding(sources, sentences, source_mask)
        encoded_chars = self._encoder(embedded_chars, char_encoder_mask)

        encoded_words = self._encoder_sentence(embedded_words, word_encoder_mask)
        prediction_tags = self._tag_classifier(encoded_words)[target_tags != MorphoDataset.Factor.PAD, :]
        target_tags = target_tags[target_tags != MorphoDataset.Factor.PAD]

        hooked_encoded_words = 1.0*encoded_words
        hooked_encoded_words.register_hook(lambda g: g / (sentence_lengths.float().sqrt().view(-1,1,1) + 1e-9))
        encoded_words = torch.repeat_interleave(hooked_encoded_words, sentence_lengths, dim=0)
        word_encoder_mask = torch.repeat_interleave(word_encoder_mask, sentence_lengths, dim=0)

        prediction_lemmas = self._decoder(encoded_chars, encoded_words, targets_in, decoder_combined_mask, char_encoder_mask, word_encoder_mask, sentence_lengths)
        prediction_lemmas = prediction_lemmas[targets_out != MorphoDataset.Factor.PAD]
        targets_out = targets_out[targets_out != MorphoDataset.Factor.PAD]

        return prediction_lemmas, prediction_tags, targets_out, target_tags

    def predict(self, batch, dataset):
        sources, source_mask, targets, sentences, char_encoder_mask, word_encoder_mask, sentence_lengths, target_tags = self._gather_batch(batch, dataset)
        maximum_iterations = sources.size(1) + 10

        sources, embedded_chars, embedded_words = self._input_embedding(sources, sentences, source_mask)
        encoded_chars = self._encoder(embedded_chars, char_encoder_mask)

        encoded_words = self._encoder_sentence(embedded_words, word_encoder_mask)
        prediction_tags = self._tag_classifier(encoded_words)[target_tags != MorphoDataset.Factor.PAD, :]
        target_tags = target_tags[target_tags != MorphoDataset.Factor.PAD]

        encoded_words = torch.repeat_interleave(encoded_words, sentence_lengths, dim=0)
        word_encoder_mask = torch.repeat_interleave(word_encoder_mask, sentence_lengths, dim=0)

        output = torch.full((encoded_chars.size(0), 1), MorphoDataset.Factor.BOW, device='cuda', dtype=torch.long)
        finished = torch.full((encoded_chars.size(0),), False, device='cuda', dtype=torch.uint8)

        for _ in range(maximum_iterations):
            decoder_combined_mask = self._create_look_ahead_mask(output)
            predictions = self._decoder(encoded_chars, encoded_words, output, decoder_combined_mask, char_encoder_mask, word_encoder_mask, sentence_lengths)

            next_prediction = predictions[:, -1, :]
            next_char = next_prediction.argmax(dim=-1)

            output = torch.cat((output, next_char.unsqueeze(-1)), -1)
            finished |= next_char == MorphoDataset.Factor.EOW

            if finished.all(): break

        return output[:, 1:], prediction_tags, sources, targets, target_tags, sentence_lengths

    def predict_to_list(self, batch, dataset):
        lemma_sentences = []
        tag_sentences = []
        
        predictions, prediction_tags, _, _, _, sentence_lengths = self.predict(batch, dataset)
        predictions = predictions.cpu()
        prediction_tags = torch.argmax(prediction_tags.data, 1).cpu()

        index = 0
        for s, length in enumerate(sentence_lengths):
            sentence = []
            tags = []
            for w in range(length.item()):
                word = []
                for prediction in predictions[index, :].numpy():
                    if prediction == MorphoDataset.Factor.EOW: break
                    word.append(prediction)
                    
                word.append(MorphoDataset.Factor.EOW)
                tags.append(prediction_tags[index].item())
                sentence.append(word)

                index += 1

            lemma_sentences.append(sentence)
            tag_sentences.append(tags)

        return lemma_sentences, tag_sentences