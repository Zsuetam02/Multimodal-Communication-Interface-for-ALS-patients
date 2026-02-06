import torch
import torch.nn as nn
import torchvision.models as models


#  EOG LSTM BRANCH
class EOGNet(nn.Module):
    def __init__(self, num_classes=5, feature_dim=14, hidden_size=64, lstm_layers=2):
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        lstm_out_dim = hidden_size * 2

        # MLP
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Final embedding
        self.out_dim = lstm_out_dim + 32

    def forward(self, x_seq, x_feat):
        """
        x_seq: (B, 2, T) → transpose → (B, T, 2)
        x_feat: (B, feature_dim)
        """
        x_seq = x_seq.transpose(1, 2)

        lstm_out, _ = self.lstm(x_seq)
        seq_embed = lstm_out[:, -1, :]

        feat_embed = self.feature_fc(x_feat)

        return torch.cat([seq_embed, feat_embed], dim=1)



#  VIDEO RESNET BRANCH
class VideoNet(nn.Module):
    def __init__(self, pretrained=True, in_ch=1):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # replace first conv for grayscale
        if in_ch != 3:
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
        else:
            self.features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


#  FUSION NET

class FusionNet(nn.Module):
    def __init__(self, num_classes=5, mode="fusion", in_ch=1,
                 feature_dim=14, hidden_size=64, lstm_layers=2):
        """
        mode:
            "video"  – only video branch
            "eog"    – only EOG branch
            "fusion" – both modalities fused
        """
        super().__init__()
        self.mode = mode

        # Branches
        self.eog_net = EOGNet(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers
        )
        self.video_net = VideoNet(pretrained=True, in_ch=in_ch)

        # Classifier
        if mode == "video":
            fusion_in = self.video_net.out_dim
        elif mode == "eog":
            fusion_in = self.eog_net.out_dim
        else:
            fusion_in = self.video_net.out_dim + self.eog_net.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    # FORWARD
    def forward(self, frame=None, x_seq=None, x_feat=None):

        if self.mode == "video":
            video_embed = self.video_net(frame)
            return self.classifier(video_embed)

        if self.mode == "eog":
            eog_embed = self.eog_net(x_seq, x_feat)
            return self.classifier(eog_embed)

        # fusion mode
        video_embed = self.video_net(frame)
        eog_embed = self.eog_net(x_seq, x_feat)

        fused = torch.cat([video_embed, eog_embed], dim=1)
        return self.classifier(fused)
