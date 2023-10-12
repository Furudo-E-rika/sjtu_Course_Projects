import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeNN(nn.Module):
    def __init__(self, config):
        super(CodeNN, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.embedding = nn.Embedding(config['n_words'], config['emb_size'])
        self.name_encoder = nn.LSTM(input_size=config['emb_size'], hidden_size=config['lstm_dims'], bidirectional=True, batch_first=True)
        self.desc_encoder = nn.LSTM(input_size=config['emb_size'], hidden_size=config['lstm_dims'], bidirectional=True, batch_first=True)
        self.linear_name = nn.Linear(config['lstm_dims'] * 2, config['n_hidden'])
        self.linear_token = nn.Linear(config['emb_size'], config['n_hidden'])
        self.linear_desc = nn.Linear(config['lstm_dims'] * 2, config['n_hidden'])
        self.linear_fuse = nn.Linear(config['n_hidden'] * 2, config['n_hidden'])

    def code_encoding(self, name, name_len, tokens, tok_len):

        ## name encoding

        batch_size, name_seq_len = name.size()
        name_inputs = self.embedding(name)
        name_inputs = F.dropout(name_inputs, p=0.25, training=self.training)


        name_input_lens_sorted, name_indices = torch.sort(name_len, descending=True)
        name_inputs_sorted = name_inputs.index_select(0, name_indices)
        name_inputs_packed = pack_padded_sequence(name_inputs_sorted, name_input_lens_sorted.cpu().data.tolist(), batch_first=True)

        name_hidden, (name_h, name_c) = self.name_encoder(name_inputs_packed)

        _, name_indices_recover = torch.sort(name_indices)
        name_hidden, lens = pad_packed_sequence(name_hidden, batch_first=True)
        name_hidden = F.dropout(name_hidden.data, p=0.25, training=self.training)
        name_hidden = name_hidden.index_select(0, name_indices_recover)
        # name_h = name_h.index_select(1, name_indices_recover)

        pooled_name_encoding = F.max_pool1d(name_hidden.transpose(1, 2), name_seq_len).squeeze(2)

        name_encoding = torch.tanh(pooled_name_encoding)

        ## token encoding
        batch_size, tokens_seq_len = tokens.size()
        token_inputs = self.embedding(tokens)
        token_inputs = F.dropout(token_inputs, p=0.25, training=self.training)

        pooled_token_encoding = F.max_pool1d(token_inputs.transpose(1, 2), tokens_seq_len).squeeze(2)
        token_encoding = torch.tanh(pooled_token_encoding)

        name_encoding = self.linear_name(name_encoding)
        token_encoding = self.linear_token(token_encoding)
        fuse_encoding = torch.tanh(torch.concat([name_encoding, token_encoding], dim=1))
        code_repr = self.linear_fuse(fuse_encoding)

        return code_repr


    def desc_encoding(self, desc, desc_len):

        batch_size, desc_seq_len = desc.size()
        desc_inputs = self.embedding(desc)
        desc_inputs = F.dropout(desc_inputs, p=0.25, training=self.training)

        desc_input_lens_sorted, desc_indices = torch.sort(desc_len, descending=True)
        desc_inputs_sorted = desc_inputs.index_select(0, desc_indices)
        desc_inputs_packed = pack_padded_sequence(desc_inputs_sorted, desc_input_lens_sorted.cpu().data.tolist(), batch_first=True)

        desc_hidden, (desc_h, desc_c) = self.desc_encoder(desc_inputs_packed)

        _, desc_indices_recover = torch.sort(desc_indices)
        desc_hidden, lens = pad_packed_sequence(desc_hidden, batch_first=True)
        desc_hidden = F.dropout(desc_hidden.data, p=0.25, training=self.training)
        desc_hidden = desc_hidden.index_select(0, desc_indices_recover)

        pooled_desc_encoding = F.max_pool1d(desc_hidden.transpose(1, 2), desc_seq_len).squeeze(2)
        desc_encoding = torch.tanh(pooled_desc_encoding)

        desc_repr = self.linear_desc(desc_encoding)
        return desc_repr

    def similarity(self, code_vec, desc_vec):
        return F.cosine_similarity(code_vec, desc_vec)

    def forward(self, name, name_len, tokens, tok_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        code_repr = self.code_encoding(name, name_len, tokens, tok_len)
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr)  # [batch_sz x 1]

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss
