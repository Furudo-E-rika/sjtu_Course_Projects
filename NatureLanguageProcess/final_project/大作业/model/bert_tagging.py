import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
MAX_LENGTH = 40

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.bert_model = AutoModel.from_pretrained(config.pretrain)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.pretrain)
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        lengths = batch.lengths
        utt = batch.utt
        device = batch.device

        input_ids = []
        input_mask = []
        tag_ids_list = tag_ids.tolist()
        tag_mask_list = tag_mask.tolist()
        tag_ids = []
        tag_masks = []

        for i in range(len(lengths)):
            token = self.bert_tokenizer.tokenize(utt[i])
            ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + token + ["[SEP]"])
            mask = [1] * len(ids)
            mask = mask + [0] * (MAX_LENGTH - len(ids))
            ids = ids + [0] * (MAX_LENGTH - len(ids))
            input_ids.append(ids)
            input_mask.append(mask)

            tag_id = [0] + tag_ids_list[i] + [0] * (MAX_LENGTH-len(tag_ids_list[i])-1)
            tag_mask = [0] + tag_ids_list[i] + [0] * (MAX_LENGTH - len(tag_mask_list[i]) - 1)
            tag_ids.append(tag_id)
            tag_masks.append(tag_mask)

        input_ids = torch.LongTensor(input_ids).to(device)
        input_mask = torch.LongTensor(input_mask).to(device)
        tag_ids = torch.LongTensor(tag_ids).to(device)
        tag_masks = torch.LongTensor(tag_masks).to(device)

        bert_output = self.bert_model(input_ids, attention_mask=input_mask)
        last_hidden_state = bert_output.last_hidden_state
        rnn_out, h_t_c_t = self.rnn(last_hidden_state)

        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_masks, tag_ids)

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

