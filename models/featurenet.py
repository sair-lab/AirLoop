#!/usr/bin/env python3

from functools import partial

import numpy as np

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


class AttnFC(nn.Module):
    def __init__(self, feat_dim, desc_dim, emb_dim=128, feat_num=300, n_heads=4, n_pass=4, drop=0.1):
        super().__init__()
        self.attn = nn.Sequential(*[
            TransformerLayer(feat_dim, emb_dim, n_heads=n_heads)
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


class FPNEncoder(nn.Module):
    def __init__(self, model, ckpts):
        super().__init__()
        self.encoder = model
        module_dict = dict(self.encoder.named_modules())
        for (name, _, _) in ckpts:
            module_dict[name].register_forward_hook(partial(self._save_feature, name))
        self._saved_features = {}

    def forward(self, x):
        _saved_features = self._saved_features.setdefault(x.device, {})

        self.encoder(x)
        features = _saved_features.copy()
        _saved_features.clear()
        return features

    def _save_feature(self, name, mod, inp, out):
        self._saved_features[inp[0].device][name] = out


class FPNDecoder(nn.Module):
    def __init__(self, ckpts, res, out_channel):
        super().__init__()
        self.ckpts = list(reversed(ckpts))
        res = np.array(res)

        self.decoder = nn.ModuleList()
        for i, ((_, in_chan, in_sc), (_, out_chan, out_sc)) in enumerate(zip(self.ckpts[:-1], self.ckpts[1:])):
            cat_chan = 0 if i == 0 else self.ckpts[i][1]
            in_res, out_res = np.ceil(res / in_sc).astype(int), np.ceil(res / out_sc).astype(int)
            out_pad = out_res - 1 - (in_res - 1) * 2
            self.decoder.append(nn.Sequential(
                *(make_layer(in_chan + cat_chan, out_chan, bn=nn.GroupNorm(min(out_chan // 4, 32), out_chan)) +
                  make_layer(out_chan, out_chan, conv=nn.ConvTranspose2d, stride=2, output_padding=tuple(out_pad), bn=nn.GroupNorm(min(out_chan // 4, 32), out_chan)))
            ))
        self.decoder.append(nn.Sequential(
            *make_layer(self.ckpts[-1][1] * 2, out_channel, kernel_size=1, padding=0, activation=nn.Identity())))

    def forward(self, features):
        fp = [features[ckpt] for ckpt, _, _ in self.ckpts]

        b, _, h, w = fp[0].shape
        x = torch.zeros(b, 0, h, w).to(fp[0])
        for dec, f in zip(self.decoder, fp):
            x = dec(torch.cat([x, f], dim=1))
        return x


class FeatureNet_(nn.Module):
    CKPTS = [
        ('features.3', 64, 1),
        ('features.8', 128, 2),
        ('features.17', 256, 4),
        ('features.26', 512, 8),
        ('features.35', 512, 16),
    ]

    def __init__(self, n_features, res, feat_dim):
        super().__init__()
        self.n_features = n_features

        # remove extra modules and hook up full-scale feature conv
        vgg = models.vgg19(pretrained=True)
        vgg.avgpool = vgg.classifier = nn.Identity()

        self.encoder = FPNEncoder(vgg, self.CKPTS)
        self.decoder = FPNDecoder(self.CKPTS, res, 1)
        self.trans = nn.ModuleList([nn.Sequential(
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
            *make_layer(ch, ch, bn=nn.GroupNorm(min(ch // 4, 32), ch)),
        ) for (_, ch, _) in self.CKPTS])
        self.proj = nn.Sequential(
            nn.Linear(sum(ckpt[1] for ckpt in self.CKPTS), 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        self.const_boarder = ConstantBorder(border=4, value=0)

        self.grid_sample = GridSample()

    def forward(self, img):
        b, _, h, w = img.shape

        feature_maps_orig = self.encoder(img)
        score_map = self.decoder(feature_maps_orig)

        feature_maps = [trans(feature_maps_orig[ckpt[0]]) for ckpt, trans in zip(self.CKPTS, self.trans)]
        score_map = self.softnms(score_map, (5, 5))

        scores, points = self.const_boarder(nms.nms2d(score_map, (5, 5))).view(b, -1, 1).topk(self.n_features, dim=1)
        points = torch.cat((points % w, points // w), dim=-1)
        points = C.normalize_pixel_coordinates(points, h, w)

        raw_features = torch.cat([self.grid_sample((feature, points)) for feature in feature_maps], dim=-1)
        return points, score_map, scores, self.proj(raw_features), raw_features, feature_maps_orig

    def softnms(self, x, ks=(3, 3)):
        b, c, h, w = x.shape
        numel = ks[0] * ks[1]
        kernel = torch.eye(numel).reshape(numel, 1, *ks).repeat(c, 1, 1, 1).to(x)
        pad = ((ks[1] - 1) // 2, (ks[1] - 1) // 2, (ks[0] - 1) // 2, (ks[0] - 1) // 2)
        exp_x = torch.exp(x)
        unfolded = F.conv2d(F.pad(exp_x, pad, mode='constant', value=0), kernel, stride=1, groups=c).reshape(b, c, -1, h, w)
        # trashbin
        unfolded = torch.cat([unfolded, torch.ones(b, c, 1, h, w).to(unfolded)], dim=2)
        return torch.sigmoid(x) * exp_x / unfolded.sum(dim=2)



class FeatureNet(models.VGG):
    def __init__(self, res=(240, 320), feat_dim=256, feat_num=500, gd_dim=1024, sample_pass=0):
        super().__init__(models.vgg13().features)
        self.feat_dim, self.feat_num, self.sample_pass = feat_dim, feat_num, sample_pass
        self.features = FeatureNet_(self.feat_num, res, feat_dim)
        self.gd_indim = self.features.CKPTS[-1][1]
        self.global_desc = GeM(self.gd_indim, gd_dim)
        # self.global_desc = AttnFC(self.gd_indim, gd_dim, emb_dim=256, n_pass=2)
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
        self.pos_enc_gd = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, self.gd_indim))
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

        points, pointness, scores, descriptors, raw_features, feature_maps = self.features(inputs)

        # fea = features.permute(0, 2, 3, 1).reshape(B, -1, self.feat_dim)

        # descriptors = descriptors + self.pos_enc(points)
        # descriptors = self.graph(descriptors)

        fea = feature_maps[self.features.CKPTS[-1][0]]
        gd, gd_locs = self.global_desc(fea.reshape(B, self.gd_indim, fea.shape[-1] * fea.shape[-2]).transpose(-1, -2))
        # fea = fea[:, ::4]

        N = self.feat_num
        return descriptors.reshape(B, N, self.feat_dim), points.reshape(B, N, 2), pointness, scores.reshape(B, N), ((gd, gd_locs) if self.training else gd), None
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


def make_layer(in_chan, out_chan, kernel_size=3, stride=1, padding=1, conv=nn.Conv2d, bn=None, activation=nn.ReLU(), **kwargs):
    modules = [conv(in_chan, out_chan, kernel_size, stride, padding, **kwargs)] + \
        ([bn] if bn is not None else []) + \
        ([activation] if activation is not None else [])
    return modules


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
