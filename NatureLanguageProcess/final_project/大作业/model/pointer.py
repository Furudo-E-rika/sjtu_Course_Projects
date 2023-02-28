# coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from model.attention import Attention
from copy import copy
from torch.nn import Parameter
from model.attention import PointerAttention

MAX_LENGTH = 50


class PointerTagging(nn.Module):

    def __init__(self, config):
        super(PointerTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size, num_layers=4, batch_first=True)

        self.decoder = PointerDecoder(config.embed_size, config.hidden_size, config.num_tags)
        self.sm = nn.Softmax(dim=-1)

        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.decoder_input0 = Parameter(torch.FloatTensor(config.embed_size), requires_grad=False)
        nn.init.uniform(self.decoder_input0, -1, 1)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        batch_size = len(batch)

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        encoder_output, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)

        decoder_hidden0 = (h_t_c_t[0][-1], h_t_c_t[1][-1])
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        (outputs, pointers), decoder_hidden = self.decoder(embed, decoder_input0, decoder_hidden0, encoder_output)

        tag_output = self.output_layer(encoder_output, (outputs, pointers), tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0].view(batch_size, -1, output[0].shape[1])
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


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_tags):
        super(PointerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_to_hidden = nn.Linear(embed_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = PointerAttention(hidden_dim, hidden_dim)
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
        self.num_tags = num_tags

    def forward(self, embed_input, decoder_input, hidden, context):

        batch_size = embed_input.size(0)
        input_length = embed_input.size(1)

        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.attn.init_inf(mask.size())
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            h, c = hidden
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)  # batch * hidden_size

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            hidden_t, output = self.attn(h_t, context, torch.eq(mask, 0))

            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        for _ in range(input_length):

            h_t, c_t, outs = step(decoder_input, hidden)

            hidden = (h_t, c_t)

            masked_outs = outs * mask

            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embed_dim).byte()
            decoder_input = embed_input[embedding_mask.data].view(batch_size, self.embed_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 2, 0)  # batch_size * seq_len * seq_len

        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden

class TaggingFNNDecoder(nn.Module):
    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.embed_layer = nn.Linear(input_size, 2*MAX_LENGTH)
        self.output_layer = nn.Linear(3*MAX_LENGTH, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, rnn_out, pointer_output, mask, labels=None):

        #print(rnn_out.shape)

        rnn_out = self.embed_layer(rnn_out)
        (ptr_out, ptr) = pointer_output
        batch_size, seq_len, _ = ptr_out.shape
        extra_zeros = torch.zeros((batch_size, seq_len, MAX_LENGTH-seq_len)).to(ptr_out.device)

        input = torch.cat([ptr_out, extra_zeros, rnn_out], dim=-1)
        #print(input.shape, ptr_out.shape, extra_zeros.shape, rnn_out.shape)
        logits = self.output_layer(input)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32

        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            #loss = self.loss_fct(logits, labels.view(-1))

            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob,)

