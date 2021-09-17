from dataclasses import dataclass, field
from typing import Any, Dict, List

import kornia.geometry.conversions as C
import numpy as np
import os
import pickle, bz2
import torch
import tqdm

from prettytable import PrettyTable, MARKDOWN

from .geometry import Projector, gen_probe, feature_pt_ncovis
from .misc import GlobalStepCounter
from .utils import ConsecutiveMatch, GridSample, PairwiseCosine


class MatchEvaluator():
    def __init__(self, back=[1], viz=None, viz_dist_min=0, viz_dist_max=100, top=None, writer=None, counter=None, args=None):
        self.back, self.top = back, top
        self.viz, self.viz_min, self.viz_max = viz, viz_dist_min, viz_dist_max
        self.hist = []
        self.grid_sample = GridSample()
        self.match = ConsecutiveMatch()
        self.error = {b: [] for b in back}
        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.cur_env = None
        self.env_err_seg = {}
        self.args = args

    @torch.no_grad()
    def observe(self, descriptors, points, scores, gd, score_map, aux, imgs, env_seq):
        depth_map, poses, Ks = aux
        B, N, _ = points.shape
        _, _, H, W = imgs.shape
        top = N if self.top is None else self.top

        env_seq = "_".join(zip(*env_seq).__next__())
        if self.cur_env != env_seq:
            last_env, self.cur_env = self.cur_env, env_seq
            self.hist = []
            n_batches = len(self.error[self.back[0]])
            self.env_err_seg[self.cur_env] = [n_batches, -1]
            if last_env is not None:
                self.env_err_seg[last_env][1] = n_batches

        # populate hist until sufficient
        depths = self.grid_sample((depth_map, points)).squeeze(-1)
        for img, desc, pt, pose, K, depth in zip(imgs, descriptors, points, poses, Ks, depths):
            self.hist.insert(0, (img, desc, pt, pose, K, depth))
        if len(self.hist) - B < max(self.back):
            return

        viz_img1, viz_img2, viz_pts1, viz_pts2, viz_val = [], [], [], [], []
        imgs_new, descs_new, pts_new, poses_new, Ks_new, depths_new = self._unpack_hist(reversed(self.hist[:B]))
        for b in self.back:
            imgs_old, descs_old, pts_old, poses_old, Ks_old, depths_old = self._unpack_hist(reversed(self.hist[b:b+B]))
            matched_new_pt, conf = self.match(descs_old, descs_new, pts_new)
            match_conf, top_idx = conf.topk(top)
            match_src, match_dst = self._point_select(pts_old, top_idx), self._point_select(matched_new_pt, top_idx)
            depths_old = self._point_select(depths_old, top_idx)
            error = reproj_error(match_src, Ks_old, poses_old, depths_old, match_dst, Ks_new, poses_new, H, W)
            self.error[b].append(error)
            if self.viz is not None:
                viz_img1.append(imgs_old), viz_img2.append(imgs_new)
                viz_pts1.append(match_src), viz_pts2.append(match_dst)
                viz_val.append(error)

        if self.viz is not None:
            # B, back, *
            viz_img1, viz_img2 = torch.stack(viz_img1, dim=1), torch.stack(viz_img2, dim=1)
            viz_pts1, viz_pts2 = torch.stack(viz_pts1, dim=1), torch.stack(viz_pts2, dim=1)
            viz_val = torch.stack(viz_val, dim=1).detach().cpu().numpy()
            for img1, img2, pts1, pts2, val in zip(viz_img1, viz_img2, viz_pts1, viz_pts2, viz_val):
                self.viz.showmatch(img1, pts1, img2, pts2, 'hot', val, self.viz_min, self.viz_max, name='backs')
                break

        self.hist = self.hist[:-B]

        return self.error

    def ave_reproj_error(self, quantile=None, env=None):
        mean, quantiles = {}, {}
        for b in self.back:
            seg = self.env_err_seg[env] if env is not None else [0, -1]
            error = torch.cat(self.error[b][seg[0]:seg[1]])
            mean[b] = error.mean().item()
            if quantile is not None:
                quantiles[b] = torch.quantile(error, torch.tensor(quantile).to(error)).tolist()
        return (mean, quantiles) if quantile is not None else mean

    def ave_prec(self, thresh=1, env=None):
        perc = {}
        for b in self.back:
            seg = self.env_err_seg[env] if env is not None else [0, -1]
            perc[b] = (torch.cat(self.error[b][seg[0]:seg[1]]) < 1).to(torch.float).mean().item()
        return perc

    def report(self, err_quant=[0.5, 0.9]):
        result = PrettyTable(['Env Name', 'n-back',
                              'Mean Err (%s)' % self._fmt_list(np.array(err_quant) * 100, '%d%%'), 'Ave Prec'])
        result.float_format['Ave Prec'] = '.2'
        result.align['Env Name'] = 'r'

        all_mean, all_quantiles = self.ave_reproj_error(quantile=err_quant)
        prec = self.ave_prec()
        result.add_rows([['All', b, '%.2f (%s)' % (all_mean[b], self._fmt_list(all_quantiles[b], '%.2f')),
                          prec[b]] for b in self.back])

        for e in self.env_err_seg:
            env_mean, env_quantiles = self.ave_reproj_error(quantile=err_quant, env=e)
            env_prec = self.ave_prec(env=e)
            result.add_rows([[e, b, '%.2f (%s)' % (env_mean[b], self._fmt_list(env_quantiles[b], '%.2f')),
                              env_prec[b]] for b in self.back])

        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_scalars('Eval/Match/MeanErr', {'%d-back' % b: v for b, v in all_mean.items()}, n_iter)
            self.writer.add_scalars('Eval/Match/AvePrec', {'%d-back' % b: v for b, v in prec.items()}, n_iter)
            for b in self.back:
                self.writer.add_histogram('Eval/Match/LogErr/%d-back' % b,
                                          torch.log10(torch.cat(self.error[b]).clamp(min=1e-10)), n_iter)
            # TensorBoard supports markdown table although column alignment is broken
            result.set_style(MARKDOWN)
            self.writer.add_text('Eval/Match/PerSeq', result.get_string(sortby='n-back'), n_iter)
        else:
            print('Evaluation: step %d' % n_iter)
            print(result.get_string(sortby='n-back'))

        if self.args.eval_save is not None:
            np.savez_compressed('%s-match-err' % self.args.eval_save, **{str(k): v for k, v in self.error.items()})
            np.savez_compressed('%s-match-seg' % self.args.eval_save, **{str(k): v for k, v in self.error.items()})

        # clear history
        self.error = {b: [] for b in self.back}
        self.hist = []
        self.cur_env = None
        self.env_err_seg = {}

    @staticmethod
    def _fmt_list(elems, fmt, delim=', '):
        return delim.join([fmt % e for e in elems])

    @staticmethod
    def _unpack_hist(hist):
        return [torch.stack(attr, dim=0) for attr in zip(*hist)]

    @staticmethod
    def _point_select(attr, idx):
        B, N = idx.shape
        return attr.gather(1, idx.view(B, N, *([1] * (len(attr.shape) - 2))).expand(B, N, *attr.shape[2:]))

    def save_error(self, file_path):
        error = {k: [v.detach().cpu().numpy() for v in vs] for k, vs in self.error.items()}
        with bz2.BZ2File(file_path, 'wb') as f:
            pickle.dump([error, self.env_err_seg], f)


def reproj_error(pts_src, K_src, pose_src, depth_src, pts_dst, K_dst, pose_dst, H, W):
    projector = Projector()
    cam_src = Projector._make_camera(H, W, K_src, pose_src)
    cam_dst = Projector._make_camera(H, W, K_dst, pose_dst)
    pts_prj, _ = Projector._project_points(pts_src, depth_src, cam_src, cam_dst)
    diff = C.denormalize_pixel_coordinates(pts_prj, H, W) - C.denormalize_pixel_coordinates(pts_dst, H, W)
    return torch.hypot(diff[..., 0], diff[..., 1])


@dataclass
class _EnvData():
    env: str
    gds: List[torch.Tensor] = field(default_factory=list)
    n_observed: int = 0
    seq_lims: Dict[str, List[int]] = field(default_factory=dict)
    aux: List[Any] = field(default_factory=list)


def chunk_index_itr(total: int, chunk_size: int):
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(total, chunk_start + chunk_size)
        yield chunk_start, chunk_end


class RecognitionEvaluator():
    def __init__(self, loader, n_feature=250, D_frame=256, ks=[2, 5, 10], viz=None, writer=None, counter=None, args=None):
        self.viz = viz
        self.probe_pts, self.gds, self.n_observed = None, None, 0
        self.adj_mat = None
        self.loader = loader
        self.n_feature, self.D_frame = n_feature, D_frame

        self.grid_sample = GridSample()
        self.projector = Projector()
        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.env_data: Dict[str, _EnvData] = {}
        self.ks = ks
        self.args = args
        self.chunk_size = 128

    @torch.no_grad()
    def observe(self, descriptors, points, scores, gd, score_map, aux, imgs, env_seq):
        B = len(imgs)
        env = env_seq[0][0]
        env_data = self.env_data.setdefault(env, _EnvData(env=env))

        env_data.gds.append(gd)

        if self.args.dataset == 'tartanair':
            seq = env_seq[1][0][0] + '_' + env_seq[1][1][0]
        elif self.args.dataset in ['nordland', 'robotcar']:
            seq = env_seq[1][0]
        # record/update sequence start/end
        seq_lims = env_data.seq_lims.setdefault(seq, [env_data.n_observed, env_data.n_observed + B])
        seq_lims[1] = env_data.n_observed + B

        if self.args.eval_gt_save is not None:
            if self.args.dataset == 'tartanair':
                depth_map, poses, Ks = aux
                probe_pts = self.projector.pix2world(gen_probe(depth_map), depth_map, poses, Ks)
                env_data.aux.extend(probe_pts)
            elif self.args.dataset == 'nordland':
                env_data.aux.extend(aux)
            elif self.args.dataset == 'robotcar':
                env_data.aux.extend(zip(*aux))

        env_data.n_observed += B

    def report(self):
        if self.args.eval_save is not None:
            cosine = PairwiseCosine()
            save_dict = {}
            for env, env_data in self.env_data.items():
                gds = torch.cat(env_data.gds)
                # reduce memory consumption
                cossim = [[cosine(gds[None, st0:nd0], gds[None, st1:nd1])[0].cpu().numpy()
                           for st1, nd1 in chunk_index_itr(len(gds), self.chunk_size)]
                          for st0, nd0 in chunk_index_itr(len(gds), self.chunk_size)]
                save_dict[env] = np.block(cossim)
            np.savez_compressed(self.args.eval_save, **save_dict)
            print(f'Saved result: {self.args.eval_save}')

        if self.args.eval_desc_save is not None:
            save_dict = {env: torch.cat(env_data.gds).cpu().numpy() for env, env_data in self.env_data.items()}
            np.savez_compressed(self.args.eval_desc_save, **save_dict)
            print(f'Saved global descriptors: {self.args.eval_desc_save}')

        if self.args.eval_gt_save is not None:
            print('Building groundtruth adjcency matrix')
            env_len, adj_mat = {}, {}
            for env, env_data in self.env_data.items():
                lims = np.array(list(env_data.seq_lims.values()))
                env_len[env] = (lims[:, 1] - lims[:, 0]).sum()
                adj_mat[env] = np.full((env_len[env], env_len[env]), np.nan, dtype=np.float32)

            env_progress = {env: 0 for env in self.env_data}
            for imgs, aux_d, env_seq in tqdm.tqdm(self.loader):
                B, env = len(imgs), env_seq[0][0]
                n_total = env_len[env]
                i = env_progress[env]
                for st, nd in chunk_index_itr(n_total, self.chunk_size):
                    adj_mat[env][st:nd, i:i+B] = self._calc_adjacency(self.env_data[env].aux[st:nd], aux_d)
                env_progress[env] += B

            os.makedirs(os.path.dirname(self.args.eval_gt_save), exist_ok=True)
            np.savez_compressed(self.args.eval_gt_save, **adj_mat)
            print(f'Saved ground truth: {self.args.eval_gt_save}')

    def _calc_adjacency(self, aux, aux_d):
        '''Calculate adjacency based on metadata'''
        if self.args.dataset == 'tartanair':
            probe_pts, (depths, poses, K) = aux, aux_d
            probe_pts = torch.stack(probe_pts)
            depths, poses, K = depths.to(probe_pts), poses.to(probe_pts), K.to(probe_pts)

            return feature_pt_ncovis(probe_pts, torch.zeros(len(K)),
                depths, poses, K, self.projector, self.grid_sample).cpu().numpy()
        elif self.args.dataset == 'nordland':
            offset, offset_d = aux, aux_d
            offset, offset_d = torch.stack(offset), offset_d

            return (1 / (np.abs(offset[:, None] - offset_d[None, :]) + 1)).cpu().numpy()
        elif self.args.dataset == 'robotcar':
            HEADING_TOL = 15
            (location, heading), (location_d, heading_d) = list(zip(*aux)), aux_d
            location, heading = torch.stack(location), torch.stack(heading)

            dist = torch.cdist(location, location_d)
            view_diff = (heading[:, None] - heading_d[None, :]).abs()
            return ((view_diff < HEADING_TOL).to(torch.float) / (dist + 1)).cpu().numpy()

    @staticmethod
    def _fmt_list(elems, fmt, delim=', '):
        return delim.join([fmt % e for e in elems])

    def get_pr_at_k(self, adj_mat, adj_mat_gt, similarity=True, k=np.arange(100) + 1):
        # relevant = lambda sim: sim == sim.max(dim=1, keepdims=True).values
        relevant = lambda sim: sim > 0.75
        def relevant(sim):
            out = torch.zeros_like(sim, dtype=torch.bool)
            return out.scatter(1, sim.topk(5, dim=1, largest=True).indices, True)

        pr = self._get_pr_single(adj_mat, adj_mat_gt, k, relevant, similarity)

    def _get_pr_single(self, pred_conf, gt_conf, at_k, relevant, similarity):
        admissible, inadmissible = (1, -1) if similarity else (0, 1e8)
        pr = {}

        for env in split_dict:
            adj_gt, adj_pred = gt_conf[env][::1], pred_conf[env][::1]
            # adj_gt = recog_eval.paint_block_diag(adj_gt, split_dict[env], inadmissible)
            # adj_pred = recog_eval.paint_block_diag(adj_pred, split_dict[env], inadmissible)
            _gt = recog_eval.paint_band_diag(adj_gt, split_dict[env], 0, 0, inadmissible)#[:100, :100]
            _pred = recog_eval.paint_band_diag(adj_pred, split_dict[env], 0, 0, inadmissible)#[:100, :100]
            pr[env] = precision_recall(torch.from_numpy(_gt).to('cuda'), torch.from_numpy(_pred).to('cuda'), at_k, relevant, similarity)
            torch.cuda.empty_cache()
        return pr


def mAP(pr_dict, weight_dict=None):
    AP = {}
    for env, pr in pr_dict.items():
        pr = np.array(pr)
        p, r = pr[np.isfinite(pr).all(axis=1)].T
        AP[env] = np.abs(np.trapz(r, p))
    if weight_dict is not None:
        return weighted(AP, weight_dict)
    return AP


def weighted(v_dict, w_dict):
    assert v_dict.keys() == w_dict.keys()
    return sum([v_dict[env] * w_dict[env] for env in w_dict]) / sum(w_dict.values())


@torch.no_grad()
def precision_recall(gt_conf, pred_conf, ks, relevant, similarity=False):
    pred_rank = pred_conf.sort(dim=1, descending=similarity).indices
    is_relevant = relevant(gt_conf)
    def _pr(k):
        _relevant = is_relevant.gather(1, pred_rank[:, :k]).to(torch.float32)
        _n_rec_rel = _relevant.sum(dim=1)
        return (_n_rec_rel / k).mean().item(), (_n_rec_rel / is_relevant.sum(dim=1).clamp(min=1e-6)).mean().item()
    
    return np.array([_pr(k) for k in ks])

