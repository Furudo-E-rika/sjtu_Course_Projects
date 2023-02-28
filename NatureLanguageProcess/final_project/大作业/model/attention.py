import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Attention(nn.Module):

    def __init__(self, type, hidden_size):
        super(Attention, self).__init__()
        self.type = type
        self.hidden_size = hidden_size
        if self.type == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.type == 'mlp':
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

        if self.type == 'mlp':
            wq = self.W(hidden.contiguous().view(-1, hidden_dim))
            wq = wq.view(batch, hidden_len, 1, dim)
            wq = wq.expand(batch, hidden_len, ctx_len, dim)

            uc = self.U(ctx.contiguous().view(-1, ctx_dim))
            uc = uc.view(batch, 1, ctx_len, dim)
            uc = uc.expand(batch, hidden_len, ctx_len, dim)

            wquc = self.tanh(wq + uc)

            attn = self.V(wquc.view(-1, dim).view(batch, hidden_len, ctx_len))

        elif self.type == 'dot':
            assert ctx_dim == hidden_dim
            ctx = ctx.transpose(1, 2)
            attn = torch.bmm(hidden, ctx)

        else:
            hidden = self.W(hidden)
            ctx = ctx.transpose(1, 2)
            attn = torch.bmm(hidden, ctx)

        return attn


class PointerAttention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(PointerAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention).__init__()
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