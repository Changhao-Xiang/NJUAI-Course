import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def prune_transcripts(transcripts, min_length=4):
    """Prunes sentences with < min_length words."""
    valid_indices = []
    for i in range(len(transcripts)):
        transcript_i = transcripts[i]
        words_i = transcript_i.split()
        if len(words_i) >= min_length:
            valid_indices.append(i)
    valid_indices = np.array(valid_indices)
    return valid_indices


def pad_wav(wav, wav_max_length, pad=0):
    """Pads audio wave sequence to be `wav_max_length` long."""
    dim = wav.shape[1]
    padded = np.zeros((wav_max_length, dim)) + pad
    if len(wav) > wav_max_length:
        wav = wav[:wav_max_length]
    length = len(wav)
    padded[:length, :] = wav
    return padded, length


def pad_transcript_label(transcript_label, transcript_max_length, pad=-1):
    """Pads transcript label sequence to be `transcript_max_length` long."""
    padded = np.zeros(transcript_max_length) + pad
    if len(transcript_label) > transcript_max_length:
        transcript_label = transcript_label[:transcript_max_length]
    length = len(transcript_label)
    padded[:length] = transcript_label
    return padded, length


def get_transcript_labels(transcripts, vocab, silent_vocab):
    """Converts transcript texts to sequences of vocab indices for characters."""
    transcript_labels = []
    num_vocab = len(vocab)
    for transcript in transcripts:
        words = transcript.split()
        labels = []
        for i in range(len(words)):
            word = words[i]
            if word in silent_vocab:
                # silent vocab builds on top of vocab
                label = silent_vocab.index(word) + num_vocab
                labels.append(label)
            else:
                chars = list(word)
                labels.extend([vocab.index(ch) for ch in chars])
            # add a space in between words
            labels.append(vocab.index(" "))
        labels = labels[:-1]  # remove last space
        transcript_labels.append(labels)
    return transcript_labels


def edit_distance(src_seq, tgt_seq):
    src_len, tgt_len = len(src_seq), len(tgt_seq)
    if src_len == 0:
        return tgt_len
    if tgt_len == 0:
        return src_len

    dist = np.zeros((src_len + 1, tgt_len + 1))
    for i in range(1, tgt_len + 1):
        dist[0, i] = dist[0, i - 1] + 1
    for i in range(1, src_len + 1):
        dist[i, 0] = dist[i - 1, 0] + 1
    for i in range(1, src_len + 1):
        for j in range(1, tgt_len + 1):
            cost = 0 if src_seq[i - 1] == tgt_seq[j - 1] else 1
            dist[i, j] = min(
                dist[i, j - 1] + 1,
                dist[i - 1, j] + 1,
                dist[i - 1, j - 1] + cost,
            )
    return dist


def get_cer_per_sample(hypotheses, hypothesis_lengths, references, reference_lengths):
    assert len(hypotheses) == len(references)
    cer = []
    for i in range(len(hypotheses)):
        if len(hypotheses[i]) > 0:
            dist_i = edit_distance(
                hypotheses[i][: hypothesis_lengths[i]], references[i][: reference_lengths[i]]
            )
            # CER divides the edit distance by the length of the true sequence
            cer.append((dist_i[-1, -1] / float(reference_lengths[i])))
        else:
            cer.append(1)  # since we predicted empty
    return np.array(cer)


def reshape_and_apply(Module, inputs):
    batch_size, maxlen, input_dim = inputs.size()
    # This line reshapes the input tensor into a 2D tensor where each row corresponds to a flattened version of the input sequence
    reshaped = inputs.contiguous().view(-1, input_dim)
    outputs = Module(reshaped)
    # Finally, the output tensor is reshaped back to the original 3D shape with dimensions
    return outputs.view(batch_size, maxlen, -1)


def label_smooth_loss(log_probs, labels, num_labels, smooth_param=0.1):
    """Cross entropy loss with a temperature that smooths the distribution."""
    # convert labels to one_hotted
    with torch.no_grad():
        batch_size, maxlen = labels.size()
        labels_onehotted = torch.zeros((batch_size, maxlen, num_labels), device=labels.device).long()
        labels_onehotted = labels_onehotted.scatter_(-1, labels.long().unsqueeze(2), 1)
        labels = labels_onehotted

    assert log_probs.size() == labels.size()
    label_lengths = torch.sum(torch.sum(labels, dim=-1), dim=-1, keepdim=True)

    smooth_labels = ((1.0 - smooth_param) * labels + (smooth_param / num_labels)) * torch.sum(
        labels, dim=-1, keepdim=True
    )

    loss = torch.sum(smooth_labels * log_probs, dim=-1)
    loss = torch.sum(loss / label_lengths, dim=-1)
    return -loss.mean()


def get_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """Connectionist Temporal Classification objective function."""
    log_probs = log_probs.contiguous()
    targets = targets.long()
    input_lengths = input_lengths.long()
    target_lengths = target_lengths.long()
    log_probs = torch.permute(log_probs, (1, 0, 2))
    ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity=True)
    return ctc_loss


def combine_h_and_c(h, c):
    """Combine the signals from RNN hidden and cell states."""
    # get batch size
    batch_size = h.size(1)
    h = h.permute(1, 0, 2).contiguous()
    c = c.permute(1, 0, 2).contiguous()
    h = h.view(batch_size, -1)
    c = c.view(batch_size, -1)
    return torch.cat([h, c], dim=1)  # just concatenate  results in shape (batch_size, h_columns + c_columns)
