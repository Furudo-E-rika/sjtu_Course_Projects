# coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden=None, cell=None):

        embedded = self.dropout(self.embedding(input))
        print('embedded shape: ', embedded.shape)
        if hidden is not None and cell is not None:
            outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            outputs, (hidden, cell) = self.rnn(embedded)
        #print(hidden.shape, cell.shape)

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, num_tags):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.num_tags = num_tags
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, num_tags)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hid_dim, output_dim)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden=None, cell=None):
        #print(input.shape)
        embedded = self.dropout(self.embedding(input))
        #print(embedded.shape)
        if hidden is not None and cell is not None:
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, (hidden, cell) = self.rnn(embedded)
        output = self.sm(self.linear(output[0]))
        #print(output.shape)
        #logits = self.fc_out(output.squeeze(0))
        #print(logits.shape)
        #logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = Encoder(config.vocab_size, config.embed_size, config.hidden_size,
                                            config.num_layer, config.dropout)
        self.decoder = Decoder(config.hidden_size, config.embed_size, config.hidden_size,
                                            config.num_layer, config.dropout, config.num_tags)

        self.output = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.CEL = nn.CrossEntropyLoss(ignore_index=config.tag_pad_idx)
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size)

    def forward(self, batch):

        #batch_size = len(batch)

        tag_ids = batch.tag_ids.transpose(0, 1)
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids.transpose(0, 1)
        lengths = batch.lengths

        int_len = input_ids.shape[0]
        tag_len = tag_ids.shape[0]

        encoder_outputs = torch.zeros(int_len, self.encoder.hid_dim)
        enhidden = encell = None

        for i in range(int_len):
            enout, enhidden, encell = self.encoder(input_ids[i], enhidden, encell)
            #print(encoder_outputs.shape, enout.shape[0])
            #encoder_outputs[i] = enout

        deint = torch.zeros_like(tag_ids[0])
        dehidden = enhidden
        decell = encell

        loss = 0
        output = []
        for i in range(tag_len):
            deout, dehidden, decell = self.decoder(deint, dehidden, decell)
            deint = deout.topk(1)[1].detach().squeeze()
            output.append(deout)
            loss += self.CEL(deout, tag_ids[i])
        output = torch.stack(output)

        return output, loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)

        # print(prob.shape)
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
