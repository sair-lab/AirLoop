from dataclasses import dataclass, field
from typing import Any, Dict, List

import kornia.geometry.conversions as C
import numpy as np
import os
import torch
import tqdm

from prettytable import PrettyTable, MARKDOWN

from .geometry import Projector, gen_probe, feature_pt_ncovis
from .misc import GlobalStepCounter
from .utils import PairwiseCosine

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

        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.env_data: Dict[str, _EnvData] = {}
        self.ks = ks
        self.args = args
        self.chunk_size = 128

    @torch.no_grad()
    def observe(self, gd, aux, imgs, env_seq):
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
                probe_pts = Projector.pix2world(gen_probe(depth_map), depth_map, poses, Ks)
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

            return feature_pt_ncovis(probe_pts, torch.zeros(len(K)), depths, poses, K).cpu().numpy()
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
