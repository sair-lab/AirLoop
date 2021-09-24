import torch.nn as nn
from torchvision import models


class GeM(nn.Module):
    def __init__(self, feat_dim, desc_dim, p=3):
        super().__init__()
        self.p = p
        self.whiten = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.LeakyReLU(),
            nn.Linear(desc_dim, desc_dim)
        )

    def forward(self, features):
        mean = (features ** self.p).mean(dim=1)
        return self.whiten(mean.sign() * mean.abs() ** (1 / self.p))


class FeatureNet(nn.Module):
    def __init__(self, gd_dim=1024):
        super().__init__()
        vgg = models.vgg19(pretrained=True)
        del vgg.avgpool, vgg.classifier, vgg.features[-1]
        self.features = vgg.features

        self.fea_dim = 512
        self.global_desc = GeM(self.fea_dim, gd_dim)

    def forward(self, img):
        fea = self.features(img)
        gd = self.global_desc(fea.reshape(fea.shape[0], fea.shape[1], -1).transpose(-1, -2))
        return gd
