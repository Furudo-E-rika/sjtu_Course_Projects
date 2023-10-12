import torch
import torch.nn as nn



class HybridFusion(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )

        self.audio_outputlayer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.video_outputlayer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

        self.fusion_outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)

        audio_output = self.audio_outputlayer(audio_emb)
        video_output = self.video_outputlayer(video_emb)

        fusion_embed = torch.cat((audio_emb, video_emb), 1)
        fusion_output = self.fusion_outputlayer(fusion_embed)

        final_output = (audio_output + video_output + fusion_output) / 3

        return final_output