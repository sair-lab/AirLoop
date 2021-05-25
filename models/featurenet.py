#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
from kornia.feature import nms
from torchvision import models
import torch.nn.functional as F
import kornia.geometry.conversions as C

from utils import GridSample, PairwiseCosine


class IndexSelect(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim, self.index = dim, index

    def forward(self, x):
        self.index = self.index.to(x.device)
        return x.index_select(self.dim, self.index)


class ConstantBorder(nn.Module):
    '''
    Set Boarders to Constant
    '''
    def __init__(self, border=4, value=-math.inf):
        super().__init__()
        self.pad1 = nn.ConstantPad2d(-border, value=value)
        self.pad2 = nn.ConstantPad2d(border, value=value)

    def forward(self, x):
        return self.pad2(self.pad1(x))


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.factor = downscale_factor

    def forward(self, x):
        (N, C, H, W), S = x.shape, self.factor
        H, W = H // S, W // S
        x = x.reshape(N, C, H, S, W, S).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(N, C * S**2, H, W)
        return x

# !full
class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.9, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.tran = nn.Linear(in_features, out_features)
        self.att1 = nn.Linear(out_features, 1)
        self.att2 = nn.Linear(out_features, 1)
        self.actv = nn.Sequential(nn.LeakyReLU(beta), nn.Softmax(dim=-1))

    def forward(self, x):
        h = self.tran(x)
        att = self.att1(h) + self.att2(h).permute(0, 2, 1)
        adj = self.actv(att.squeeze())
        return self.alpha * h + (1-self.alpha) * adj @ h


class TransformerLayer(nn.Module):
    def __init__(self, dim, emb_dim=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.k_proj = nn.Linear(dim, n_heads * emb_dim)
        self.q_proj = nn.Linear(dim, n_heads * emb_dim)
        self.v_proj = nn.Linear(dim, n_heads * emb_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(n_heads * emb_dim, dim)
        self.out_fwd = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(),
            nn.Linear(dim, dim), nn.LeakyReLU(),
            nn.Linear(dim, dim))

    def forward(self, q, k=None, v=None):
        if k is None and v is None:
            k = v = q

        b, n, d = q.shape
        # B, H, N, E
        que = self.q_proj(q.reshape(b * n, d)).reshape(b, n, self.n_heads, self.emb_dim).transpose(1, 2)
        key = self.k_proj(k.reshape(b * n, d)).reshape(b, n, self.n_heads, self.emb_dim).transpose(1, 2)
        val = self.v_proj(v.reshape(b * n, d)).reshape(b, n, self.n_heads, self.emb_dim).transpose(1, 2)

        # B, H, N, N
        logits = que @ key.transpose(-1, -2) / math.sqrt(d)
        weights = F.softmax(logits, dim=-1)

        # weights @ val: B, H, N, E
        o = (weights @ val).transpose(1, 2).reshape(b * n, self.n_heads * self.emb_dim)
        o = self.norm1(self.out_proj(o) + v.reshape(b * n, -1))
        o = self.norm2(self.out_fwd(o) + o)
        return o.reshape(b, n, d)


class FGN(nn.Module):
    def __init__(self, feat_dim, alpha=0.9, beta=0.2):
        super(FGN, self).__init__()
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.tran = nn.Linear(feat_dim, feat_dim)
        self.att1 = nn.Linear(feat_dim, 1)
        self.att2 = nn.Linear(feat_dim, 1)
        self.norm = nn.Sequential(nn.LeakyReLU(beta), nn.Softmax(-1))
        self.alpha = alpha

    def forward(self, x):
        adj = self.feature_adjacency(x)
        x_ = self.tran(torch.einsum('bde,bne->bnd', adj, x))
        return self.alpha * x + (1 - self.alpha) * x_

    def feature_adjacency(self, x):
        # w_ij = f(x_i, x_j)
        w = self.norm(self.att1(x) + self.att2(x).permute(0, 2, 1))
        return self.row_normalize(self.sgnroot(x.transpose(-1, -2) @ w @ x))

    def sgnroot(self, x):
        return x.sign()*(x.abs().sqrt().clamp(min=1e-7))

    def row_normalize(self, x):
        x = x / (x.abs().sum(-1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class BatchNorm2dwC(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.bn = nn.BatchNorm3d(1)

    def forward(self, x):
        return self.bn(x.unsqueeze(1)).squeeze(1)


class ScoreHead(nn.Module):
    def __init__(self, in_scale):
        super().__init__()
        self.scores_vgg = nn.Sequential(make_layer(256, 128), make_layer(128, 64, bn=BatchNorm2dwC))
        self.scores_img = nn.Sequential(make_layer(3, 8), make_layer(8, 16, bn=BatchNorm2dwC),
            PixelUnshuffle(downscale_factor=in_scale))
        self.combine = nn.Sequential(
            make_layer(64 + 16 * in_scale**2, in_scale**2 + 1, bn=BatchNorm2dwC, activation=nn.Softmax(dim=1)),
            IndexSelect(dim=1, index=torch.arange(in_scale**2)),
            nn.PixelShuffle(upscale_factor=in_scale),
            ConstantBorder(border=4, value=0))

    def forward(self, images, features):
        scores_vgg, scores_img = self.scores_vgg(features), self.scores_img(images)
        return self.combine(torch.cat([scores_vgg, scores_img], dim=1))


class DescriptorHead(nn.Module):
    def __init__(self, feat_dim, feat_num, sample_pass):
        super().__init__()
        self.feat_dim, self.feat_num, self.sample_pass = feat_dim, feat_num, sample_pass

        self.descriptor = nn.Sequential(
            make_layer(256, self.feat_dim),
            make_layer(self.feat_dim, self.feat_dim, bn=None, activation=None))
        self.sample = nn.Sequential(GridSample(), nn.BatchNorm1d(self.feat_num))
        self.residual = nn.Sequential(make_layer(3, 128, kernel_size=9, padding=4), make_layer(128, self.feat_dim))

    def forward(self, images, features, points, scores):
        descriptors, residual = self.descriptor(features), self.residual(images)
        n_group = 1 + self.sample_pass if self.training else 1
        descriptors, residual = _repeat_flatten(descriptors, n_group), _repeat_flatten(residual, n_group)
        descriptors = self.sample((descriptors, points)) + self.sample((residual, points))
        return descriptors


class AttnFC(nn.Module):
    def __init__(self, feat_dim, desc_dim, feat_num=300, n_heads=4, n_pass=4, drop=0.1):
        super().__init__()
        self.attn = nn.Sequential(*[
            TransformerLayer(feat_dim, n_heads=n_heads)
            for _ in range(n_pass)])
        # self.attn = nn.ModuleList([
        #     nn.MultiheadAttention(feat_dim, n_heads)
        #     for _ in range(n_pass)])
        # self.graph = nn.Sequential(
        #     FGN(feat_dim),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     FGN(feat_dim))
        self.actv = nn.LeakyReLU()
        self.weight = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim), nn.LeakyReLU(),
        )
        self.content = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim), nn.ReLU(),
            nn.Linear(desc_dim, desc_dim),
        )

    def forward(self, features):
        # for at in self.attn:
        #     features = self.actv(self._res_attend(at, features, features))
        # features = self.graph(features)
        features = self.attn(features)
        content = self.content(features)
        locs = self.weight(features)
        return (content * locs).sum(dim=1), locs

    # def _res_attend(self, attn, q_desc, kv_desc):
    #     # nn.MultiheadAttention uses axis order (N, B, D)
    #     q, kv = q_desc.permute(1, 0, 2), kv_desc.permute(1, 0, 2)
    #     return attn.forward(q, kv, kv, need_weights=False)[0].permute(1, 0, 2)


class AttnMatcher(nn.Module):
    def __init__(self, feat_dim, n_heads=4, n_pass=2):
        super().__init__()
        self.attn = nn.ModuleList([TransformerLayer(feat_dim, n_heads=n_heads) for _ in range(n_pass * 2)])
        self.cosine = PairwiseCosine()

    def forward(self, x, y):
        for self_attn, cross_attn in self.attn[0::2], self.attn[1::2]:
            x, y = x + self_attn(x), y + self_attn(y)
            x, y = x + cross_attn(x, y, y), y + cross_attn(y, x, x)

        return self.cosine(x, y)



class GeM(nn.Module):
    def __init__(self, feat_dim, desc_dim, n_heads=4, n_pass=1):
        super().__init__()
        self.p = 3
        self.whiten = nn.Sequential(
            nn.Linear(feat_dim, desc_dim), nn.LeakyReLU(),
            nn.Linear(desc_dim, desc_dim)
        )

    def forward(self, features):
        mean = (features ** self.p).mean(dim=1)
        return self.whiten(mean.sign() * mean.abs() ** (1 / self.p)), None
        # return F.normalize(self.whiten(mean.sign() * mean.abs() ** (1 / self.p)), p=2, dim=1), None


class NetVLAD(nn.Module):
    def __init__(self, D, D_sq=64, K=128, n_pass=None, feat_num=None):
        super().__init__()
        D_sq = D_sq // K
        self.squeeze = nn.Sequential(nn.Linear(D, D_sq), nn.LeakyReLU(), nn.Linear(D_sq, D_sq))
        self.assign = nn.Sequential(nn.Linear(D, K), nn.Softmax(dim=-1))
        self.centeroid = nn.Parameter(torch.randn(K, D_sq))
        self.K = K
        self.D_sq = D_sq

    def forward(self, descriptors):
        B, _, _ = descriptors.shape
        dist = self.squeeze(descriptors)[:, :, None, :] - self.centeroid[None, None, :, :]
        assign = self.assign(descriptors)
        out = torch.einsum('bnkd,bnk->bkd', dist, assign)
        # out = F.normalize(out, p=2, dim=2)
        out = out.reshape(B, self.D_sq * self.K)
        out = F.normalize(out, p=2, dim=1)
        return out, None



class FeatureNet(models.VGG):
    def __init__(self, feat_dim=256, feat_num=500, gd_dim=1024, sample_pass=0):
        super().__init__(models.vgg13().features)
        self.feat_dim, self.feat_num, self.sample_pass = feat_dim, feat_num, sample_pass
        # Only adopt the first 15 layers of pre-trained vgg13. Feature Map: (512, H/8, W/8)
        self.load_state_dict(models.vgg13(pretrained=True).state_dict())
        self.features = nn.Sequential(*list(self.features.children())[:15])
        del self.classifier

        self.scores = ScoreHead(8)
        self.descriptors = DescriptorHead(feat_dim, feat_num, sample_pass)
        self.global_desc = AttnFC(feat_dim, gd_dim, n_pass=4)
        # !full
        self.graph = nn.Identity()
        # self.graph = nn.Sequential(
        #     GraphAttn(self.feat_dim, self.feat_dim),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     GraphAttn(self.feat_dim, self.feat_dim))
        # self.graph = nn.Sequential(
        #     GraphAttn(self.feat_dim, self.feat_dim, residual=True),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     GraphAttn(self.feat_dim, self.feat_dim, residual=True),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     GraphAttn(self.feat_dim, self.feat_dim, residual=True),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     GraphAttn(self.feat_dim, self.feat_dim, residual=True),
        #     nn.BatchNorm1d(feat_num), nn.LeakyReLU(0.2),
        #     GraphAttn(self.feat_dim, self.feat_dim, residual=True),
        #     )
        # self.graph = nn.Sequential(
        #     TransformerLayer(self.feat_dim),
        #     TransformerLayer(self.feat_dim),
        #     TransformerLayer(self.feat_dim),
        #     TransformerLayer(self.feat_dim))
        self.pos_enc = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.feat_dim))
        self.nms = nms.NonMaximaSuppression2d((5, 5))
        # self.matcher = AttnMatcher(feat_dim)

    def forward(self, img=None, desc0=None, desc1=None):
        if img is not None:
            return self.feature_forward(img)
        if desc0 is not None and desc1 is not None:
            return self.match_forward(desc0, desc1)

    def match_forward(self, desc0, desc1):
        return self.matcher(desc0, desc1)

    def feature_forward(self, inputs):

        B, _, H, W = inputs.shape

        features = self.features(inputs)

        pointness = self.scores(inputs, features)

        scores, points = self.nms(pointness).view(B, -1, 1).topk(self.feat_num, dim=1)

        points = torch.cat((points % W, points // W), dim=-1)

        n_group = 1
        # if self.training:
        #     n_group += self.sample_pass
        #     scores_flat_dup = _repeat_flatten(pointness.view(B, H * W), self.sample_pass)
        #     points_rand = torch.multinomial(torch.ones_like(scores_flat_dup), self.feat_num)
        #     scores_rand = torch.gather(scores_flat_dup, 1, points_rand).unsqueeze(-1)
        #     points_rand = torch.stack((points_rand % W, points_rand // W), dim=-1)
        #     points = self._append_group(points_rand, self.sample_pass, points).reshape(B * n_group, self.feat_num, 2)
        #     scores = self._append_group(scores_rand, self.sample_pass, scores).reshape(B * n_group, self.feat_num, 1)

        points = C.normalize_pixel_coordinates(points, H, W)

        descriptors = self.descriptors(inputs, features, points, scores)

        rand_end = (n_group - 1) * B

        # fea = features.permute(0, 2, 3, 1).reshape(B, -1, self.feat_dim)

        descriptors = descriptors + self.pos_enc(points)
        descriptors = torch.cat([descriptors[:rand_end], self.graph(descriptors[rand_end:])])

        fea = descriptors[rand_end:]
        gd, gd_locs = self.global_desc(fea)
        # fea = fea[:, ::4]

        N = n_group * self.feat_num
        return descriptors.reshape(B, N, self.feat_dim), points.reshape(B, N, 2), pointness, scores.reshape(B, N), ((gd, gd_locs) if self.training else gd), fea
        # return fea[:, ::4, :], points.reshape(B, N, 2), pointness, scores.reshape(B, N), gd, gd_locs

    @staticmethod
    def _append_group(grouped_samples, sample_pass, new_group):
        """(B*S, N, *) + (B, N, *) -> (B*(S+1), N, *)"""
        BS, *_shape = grouped_samples.shape
        raveled = grouped_samples.reshape(BS // max(sample_pass, 1), sample_pass, *_shape)
        return torch.cat((raveled, new_group.unsqueeze(1)), dim=1)


    # # !dino
    # def update_teacher(self):
    #     with torch.no_grad():
    #         param_s = self.global_desc.state_dict()
    #         param_t = self.global_desc_t.state_dict()
    #         for ps, pt in zip(param_s, param_t):
    #             param_t[pt].data.copy_(param_t[pt].data * self.mom + param_s[ps].data * (1 - self.mom))

    # def switch_teacher(self):
    #     with torch.no_grad():
    #         param_s = self.global_desc.state_dict()
    #         param_t = self.global_desc_t.state_dict()
    #         for ps, pt in zip(param_s, param_t):
    #             param_t[pt].data.copy_(param_s[ps].data)
    # # !dino


def make_layer(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bn=nn.BatchNorm2d, activation=nn.ReLU()):
    modules = [nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)] + \
        ([bn(out_chan)] if bn is not None else []) + \
        ([activation] if activation is not None else [])
    return nn.Sequential(*modules)


def _repeat_flatten(x, n):
    """[B0, B1, B2, ...] -> [B0, B0, ..., B1, B1, ..., B2, B2, ...]"""
    shape = x.shape
    return x.unsqueeze(1).expand(shape[0], n, *shape[1:]).reshape(shape[0] * n, *shape[1:])


if __name__ == "__main__":
    '''Test codes'''
    import argparse
    from .tool import Timer

    parser = argparse.ArgumentParser(description='Test FeatureNet')
    parser.add_argument("--device", type=str, default='cuda', help="cuda, cuda:0, or cpu")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320, 320], help='image crop size')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = FeatureNet(512, 200).to(args.device).eval()
    inputs = torch.randn(args.batch_size, 3, *args.crop_size).to(args.device)

    timer = Timer()
    with torch.no_grad():
        for i in range(5):
            descriptors, points, pointness, scores = net(inputs)
            print('%d D: %s, P: (%s, %s), S: %s' % (i, descriptors.shape, pointness.shape, points.shape, scores.shape))
    print('time:', timer.end())
