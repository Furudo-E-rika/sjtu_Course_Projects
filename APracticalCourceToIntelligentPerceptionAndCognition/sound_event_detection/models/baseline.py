import torch
import torch.nn as nn
from torch.nn import functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)

class Crnn(nn.Module):
    def __init__(self, num_freq, num_class, hidden_size, channel_list, kernel_list):
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        super(Crnn, self).__init__()

        self.bn = nn.BatchNorm2d(1)
        self.channel_list = channel_list
        self.kernel_list = kernel_list

        self.backbone = self._build_backbone()
        self.BiGRU = nn.GRU(input_size=channel_list[-1], hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_class)

    def _build_backbone(self):
        backbone = nn.Sequential()

        def ConvBlock(in_channel, out_channel, kernel_size):
            block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size))
            return block

        assert len(self.channel_list) == len(self.kernel_list) + 1
        for i in range(len(self.channel_list) - 1):
            convblock = ConvBlock(self.channel_list[i], self.channel_list[i+1], (self.kernel_list[i], 2))
            backbone.add_module('block{}'.format(i+1), convblock)

        return backbone

    def detection(self, x):
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size, time_steps, num_class]
        _, ts, nf = x.shape
        x = self.bn(x.unsqueeze(1))  # bs, 1, ts, nf

        hidden = self.backbone(x)  # bs, 128, ts/4, nf/32(1)

        hidden = hidden.flatten(start_dim=2)  # bs, 128, ts/4
        hidden = hidden.permute((0, 2, 1))  # bs, ts/4, 128

        hidden = self.BiGRU(hidden)[0]  # bs, ts/4, 256

        output = torch.sigmoid(self.fc(hidden))  # bs, ts/4, nc
        output = F.interpolate(output.permute((0, 2, 1)), ts).permute((0, 2, 1))  # bs, ts, nc
        return output

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob,
            'frame_prob': frame_prob
        }

if __name__ == '__main__':
    model = Crnn(128, 10, 128, [1, 16, 32, 64, 128, 128], [2, 2, 1, 1, 1])
    x = torch.randn(64, 501, 64)
    output = model.detection(x)
    print(output.shape)
