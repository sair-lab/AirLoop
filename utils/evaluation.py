import kornia.geometry.conversions as C
import numpy as np
import os
import pickle, bz2
import torch
import tqdm

from prettytable import PrettyTable, MARKDOWN

from .geometry import Projector, gen_probe, feature_pt_ncovis
from .misc import GlobalStepCounter
from .utils import ConsecutiveMatch, GridSample


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


class RecognitionEvaluator():
    def __init__(self, loader, n_feature=250, D_frame=256, ks=[2, 5, 10], viz=None, writer=None, counter=None, args=None):
        self.viz = viz
        self.probe_pts, self.gds, self.n_observed = None, None, 0
        self.adj_mat = None
        self.loader = loader
        self.env_sizes = self.loader.dataset.summary().groupby('env')['size'].sum().reset_index()
        self.n_feature, self.D_frame = n_feature, D_frame

        self.grid_sample = GridSample()
        self.projector = Projector()
        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.cur_env = None
        self.env_stats = {}
        self.ks = ks
        self.args = args

    @torch.no_grad()
    def observe(self, descriptors, points, scores, gd, score_map, aux, imgs, env_seq):
        depth_map, poses, Ks = aux
        env = env_seq[0][0]
        if self.cur_env != env:
            last_env, self.cur_env = self.cur_env, env
            if last_env is not None:
                self.env_stats[last_env] = (self.probe_pts, self.gds, self.adj_mat, self.n_observed)

            # restore old context / create new
            if env in self.env_stats:
                self.probe_pts, self.gds, self.adj_mat, self.n_observed = self.env_stats[env]
            else:
                n_frames = int(self.env_sizes[self.env_sizes['env'] == env]['size'])
                self.probe_pts = torch.zeros(n_frames, 1200, 3).fill_(np.nan).to(points)
                self.gds = torch.zeros(n_frames, self.D_frame).fill_(np.nan).to(gd)
                self.adj_mat = torch.zeros(n_frames, n_frames).fill_(np.nan).to(points)
                self.n_observed = 0

        # record history
        B, _, H, W = depth_map.shape
        points = gen_probe(depth_map)
        new_n_observed = self.n_observed + B
        self.probe_pts[self.n_observed:new_n_observed] = self.projector.pix2world(points, depth_map, poses, Ks)
        self.gds[self.n_observed:new_n_observed] = gd
        self.n_observed = new_n_observed

        # fill adjacency matrix

    def report(self):
        self.env_stats[self.cur_env] = (self.probe_pts, self.gds, self.adj_mat, self.n_observed)

        if self.args.eval_save is not None:
            from utils import PairwiseCosine
            cosine = PairwiseCosine(inter_batch=True)
            np.savez_compressed(self.args.eval_save, **{env: cosine(gd[:n_observed, None], gd[:n_observed, None])[:, :, 0, 0].cpu().numpy() for env, (_, gd, _, n_observed) in self.env_stats.items()})
            # np.savez_compressed('workspace/air-slam/.cache/gd_mat_fullll_gem_abandonedfactory_abandonedfactory-night', **{env: cosine(gd[:n_observed, None], gd[:n_observed, None])[:, :, 0, 0].cpu().numpy() for env, (_, gd, _, n_observed) in self.env_stats.items()})

        result = PrettyTable(['Env Name',
                            'Recall @(%s)' % self._fmt_list(self.ks, '%d', '/'),
                            'Ave Rank top-(%s)' % self._fmt_list(self.ks, '%d', '/')])
        # result.float_format['Ave Prec'] = '.2'
        result.align['Env Name'] = 'r'

        self._build_adjacency(save_path='workspace/air-slam/.cache/adj_mat.npz')

        recall, ave_rank, size = {}, {}, []

        for env, (_, gd, adj_mat, n_observed) in self.env_stats.items():
            pred_rank = torch.cdist(gd, gd).sort(dim=1, descending=False).indices
            true_rank = adj_mat.sort(dim=1, descending=True).indices
            recall[env] = [(pred_rank[:, :k] == true_rank[:, :1]).to(torch.float).sum(dim=1).mean() for k in self.ks]
            ave_rank[env] = [true_rank.gather(1, pred_rank[:, :k]).to(torch.float).mean() for k in self.ks]
            size.append(n_observed)

        result.add_row(['All', self._fmt_list(np.average(np.array([recall[e] for e in self.env_stats]), axis=0, weights=size), '%.2f', '/')])
        result.add_rows([[e, self._fmt_list(recall[e], '%.2f', '/'), self._fmt_list(ave_rank[e], '%.2f', '/')] for e in self.env_stats])

        n_iter = self.counter.steps
        if self.writer is not None:
            # self.writer.add_scalars('Eval/Recog/Recall', {'@ %d' % b: v for b, v in recall.items()}, n_iter)
            # self.writer.add_scalars('Eval/Recog/AveRank', {'@ %d' % b: v for b, v in all_mean.items()}, n_iter)
            # TensorBoard supports markdown table although column alignment is broken
            result.set_style(MARKDOWN)
            self.writer.add_text('Eval/Match/PerSeq', result.get_string(), n_iter)
        else:
            print('Evaluation: step %d' % n_iter)
            print(result.get_string(sortby='n-back'))


    def _build_adjacency(self, load_path=None, save_path=None, chunk_size=128):
        if load_path is not None:
            try:
                adj_dict = np.load(load_path)
                for env, adj in adj_dict:
                    adj_mat, n_observed = self.env_stats[env][2:]
                    adj_mat[:n_observed, :n_observed] = adj
            except:
                pass
            else:
                return

        n_frame = {env: 0 for env in self.env_stats}
        for images, depths, poses, K, env_seq in tqdm.tqdm(self.loader):
            B, env = len(images), env_seq[0][0]
            pos0, _, adj_mat, _ = self.env_stats[env]
            depths, poses, K = depths.to(pos0), poses.to(pos0), K.to(pos0)
            n_total = len(pos0)
            i = n_frame[env]
            for chunk_start in range(0, n_total, chunk_size):
                chunk_end = min(n_total, chunk_start + chunk_size)
                adj_mat[chunk_start:chunk_end, i:i+B] = feature_pt_ncovis(pos0[chunk_start:chunk_end], torch.zeros(B), depths, poses, K, self.projector, self.grid_sample)
            n_frame[env] = n_frame[env] + B

        if save_path is not None:
            adj_dict = {env: stats[2][:stats[-1], :stats[-1]].cpu().numpy() for env, stats in self.env_stats.items()}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, **adj_dict)
        
            
    @staticmethod
    def _fmt_list(elems, fmt, delim=', '):
        return delim.join([fmt % e for e in elems])

