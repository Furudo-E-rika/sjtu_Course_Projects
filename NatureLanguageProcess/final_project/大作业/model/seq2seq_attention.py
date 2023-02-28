#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model.attention import Attention
from copy import copy
import numpy as np


class Seq2SeqTagging(nn.Module):

    def __init__(self, config):
        super(Seq2SeqTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.embed_size, num_layers=4, batch_first=True)
        
        self.rnn2 = getattr(nn, self.cell)(config.embed_size, config.embed_size, num_layers=4, batch_first=True)
        self.attn = SelfAttention(config.embed_size)
        self.layer_norm = nn.LayerNorm(config.embed_size)
        self.pointer_lin = nn.Linear(3*config.embed_size, 1)
        self.outlin1 = nn.Linear(2*config.embed_size, config.embed_size)
        self.outlin2 = nn.Linear(config.embed_size, config.vocab_size, bias=True)
        self.sm = nn.Softmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        #self.output_layer = AttentionDecoder(config.embed_size)
        self.output_layer = TaggingFNNDecoder(config.embed_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        #embed += torch.randn(embed.shape).to("cuda:0")*0.5
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        # packed_inputs = self.layer_norm(packed_inputs)
        # attn, attn_input = self.attn(packed_inputs)

        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        input_len = rnn_out.shape[1]
        rnn_out = cur_out = rnn_out[:, -1, :]
        rnn_out = rnn_out.unsqueeze(dim=1)
        cur_out = cur_out.unsqueeze(dim=1)
        for i in range(input_len):
            cur_out, h_t_c_t = self.rnn2(cur_out, h_t_c_t)
            if (i==0):
                rnn_out = cur_out
            else:
                rnn_out = torch.cat((rnn_out, cur_out), dim=1)

        norm_out = self.layer_norm(rnn_out)
        attn, attn_out = self.attn(norm_out)
        #hiddens = self.dropout_layer(attn_out)

        tag_output = self.output_layer(attn_out, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        #print(prob.shape)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):
    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=True)
        self.V = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax()
        
    def forward(self, hidden, ctx):
        align = self.score(hidden, ctx)
        attn = self.sm(align)
        return attn

    def score(self, hidden, ctx):
        hidden_batch, hidden_len, hidden_dim = hidden.shape
        ctx_batch, ctx_len, ctx_dim = ctx.shape
        dim = hidden_dim
        batch = hidden_batch

        wq = self.W(hidden.contiguous().view(-1, hidden_dim))
        wq = wq.view(batch, hidden_len, 1, dim)
        wq = wq.expand(batch, hidden_len, ctx_len, dim)

        uc = self.U(ctx.contiguous().view(-1, ctx_dim))
        uc = uc.view(batch, 1, ctx_len, dim)
        uc = uc.expand(batch, hidden_len, ctx_len, dim)

        wquc = self.tanh(wq + uc)

        attn = self.V(wquc.view(-1, dim).view(batch, hidden_len, ctx_len))
        return attn


class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention / (self.d_model ** 0.5)
        attention = self.dropout(attention)
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, value)
        return attention, out



# for i in range(input_ids.shape[0]):
#     index = input_ids.shape[1]
#     for j in range(input_ids.shape[1]):
#         if (input_ids[i][j]==0):
#             index = j
#             break
#     cpy = copy(input_ids[i])
#     for j in range(index):
#         input_ids[i][j] = cpy[index-j-1]