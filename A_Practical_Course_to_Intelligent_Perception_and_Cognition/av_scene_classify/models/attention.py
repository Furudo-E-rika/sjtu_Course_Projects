import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden_states, encoder_outputs):

        # hidden_states: [1, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, length, hidden_dim]

        batch_size, length, _ = encoder_outputs.size()
        hidden_states = hidden_states.repeat(length, 1, 1).transpose(0, 1)
        input = torch.cat((encoder_outputs, hidden_states), 2).view(-1, self.dim * 2)
        output = self.fc2(torch.tanh(self.fc1(input))).view(batch_size, length)
        alpha = torch.softmax(output, dim=1)
        # print("alpha_size: ", alpha.size())
        # print("encoder_outputs_size: ", encoder_outputs.size())
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)
        # print(context.size())
        return context, alpha

class MultiModelAttention(nn.Module):

    def __init__(self, num_classes, feat_dim=512, hidden_dim=512, n_layers=1) -> None:
        super(MultiModelAttention, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.video_rnn_encoder = nn.LSTM(self.feat_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.audio_rnn_encoder = nn.LSTM(self.feat_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.video_attention = Attention(self.hidden_dim)
        self.audio_attention = Attention(self.hidden_dim)
        self.multi_attention = Attention(2 * self.hidden_dim)
        self.decoder = nn.LSTM(2 * self.hidden_dim,  2 * self.hidden_dim, self.n_layers, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]

        batch_size, time_steps, _ = audio_feat.size()

        video_encoder_output, (video_hidden_state, video_cell_state) = self.video_rnn_encoder(video_feat)
        audio_encoder_output, (audio_hidden_state, audio_cell_state) = self.audio_rnn_encoder(audio_feat)
        # print(video_hidden_state.size(), audio_hidden_state.size())
        fusion_hidden = torch.cat((video_hidden_state, audio_hidden_state), dim=2)
        fusion_cell = torch.cat((video_cell_state, audio_cell_state), dim=2)
        fusion_output = torch.cat((video_encoder_output, audio_encoder_output), dim=2)

        fusion_context, fusion_alpha = self.multi_attention(fusion_hidden, fusion_output)

        decoder_output, (decoder_hidden, decoder_cell) = self.decoder(fusion_context, (fusion_hidden, fusion_cell))
        output = self.output_layer(decoder_output).squeeze(1)

        return output

if __name__ == '__main__':
    video_feat = torch.randn(32, 100, 512)
    audio_feat = torch.randn(32, 100, 512)
    model = MultiModelAttention(10)
    output = model(audio_feat, video_feat)
    print(output.size())
