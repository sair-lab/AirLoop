from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from prettytable import PrettyTable
import torch
import tqdm

from .geometry import Projector, gen_probe, feature_pt_ncovis
from .misc import GlobalStepCounter
from .utils import PairwiseCosine


class RecognitionEvaluator():
    def __init__(self, loader, viz=None, writer=None, counter=None, args=None):
        self.viz = viz
        self.probe_pts, self.gds, self.n_observed = None, None, 0
        self.adj_mat = None
        self.loader = loader

        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.env_data: Dict[str, _EnvData] = {}
        self.args = args
        self.chunk_size = 128

        self.gt_path = None
        self.eval_gt_dir = args.eval_gt_dir
        if args.eval_gt_dir is not None:
            self.gt_path = Path(args.eval_gt_dir) / f'{args.dataset_name}.npz'

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

        if self.gt_path is not None:
            if self.args.dataset == 'tartanair':
                depth_map, poses, Ks = aux
                probe_pts = Projector.pix2world(gen_probe(depth_map), depth_map, poses, Ks)
                env_data.aux.extend(probe_pts)
            elif self.args.dataset == 'nordland':
                env_data.aux.extend(aux)
            elif self.args.dataset == 'robotcar':
                env_data.aux.extend(zip(*aux))

        env_data.n_observed += B

    @torch.no_grad()
    def report(self):
        cosine = PairwiseCosine()
        sim_dict = {}
        for env, env_data in self.env_data.items():
            gds = torch.cat(env_data.gds)
            # reduce memory consumption
            cossim = [[cosine(gds[None, st0:nd0], gds[None, st1:nd1])[0].cpu().numpy()
                        for st1, nd1 in chunk_index_itr(len(gds), self.chunk_size)]
                        for st0, nd0 in chunk_index_itr(len(gds), self.chunk_size)]
            sim_dict[env] = np.block(cossim)

        if self.args.eval_save is not None:
            np.savez_compressed(self.args.eval_save, **sim_dict)
            print(f'Saved result: {self.args.eval_save}')

        if self.args.eval_desc_save is not None:
            desc_dict = {env: torch.cat(env_data.gds).cpu().numpy() for env, env_data in self.env_data.items()}
            np.savez_compressed(self.args.eval_desc_save, **desc_dict)
            print(f'Saved global descriptors: {self.args.eval_desc_save}')

        # load or compile groundtruth adjacency
        gt_adj = None
        if self.gt_path.is_file():
            gt_adj = np.load(self.gt_path)
            print(f'Loaded ground truth: {self.gt_path}')
        elif self.eval_gt_dir is not None:
            print('Building groundtruth adjcency matrix')
            env_len, gt_adj = {}, {}
            for env, env_data in self.env_data.items():
                lims = np.array(list(env_data.seq_lims.values()))
                env_len[env] = (lims[:, 1] - lims[:, 0]).sum()
                gt_adj[env] = np.full((env_len[env], env_len[env]), np.nan, dtype=np.float32)

            env_progress = {env: 0 for env in self.env_data}
            for imgs, aux_d, env_seq in tqdm.tqdm(self.loader):
                B, env = len(imgs), env_seq[0][0]
                n_total = env_len[env]
                i = env_progress[env]
                for st, nd in chunk_index_itr(n_total, self.chunk_size):
                    gt_adj[env][st:nd, i:i+B] = self._calc_adjacency(self.env_data[env].aux[st:nd], aux_d)
                env_progress[env] += B

            os.makedirs(self.eval_gt_dir, exist_ok=True)
            np.savez_compressed(self.gt_path, **gt_adj)
            print(f'Saved ground truth: {self.gt_path}')

        if gt_adj is not None:
            table = PrettyTable(field_names=['', 'R@100P'], float_format='.3')
            criterion = get_criterion(self.args.dataset)
            for env in sim_dict:
                gt_adj_ = torch.from_numpy(gt_adj[env]).to(self.args.device).fill_diagonal_(0)
                cossim = torch.from_numpy(sim_dict[env]).to(self.args.device).fill_diagonal_(0)
                r100p = recall_at_100precision(gt_adj_, cossim, criterion)
                table.add_row([env, r100p])
            print(table.get_string())

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


def get_criterion(dataset_name):
    if dataset_name == 'tartanair':
        return lambda sim: sim > 0.5
    elif dataset_name == 'nordland':
        return lambda sim: sim > 1 / 3.5
    elif dataset_name == 'robotcar':
        return lambda sim: sim > 1 / (10 + 1)


@torch.no_grad()
def recall_at_100precision(gt_sim, pred_sim, relevant):
    pred_rank = pred_sim.sort(dim=1, descending=True).indices
    is_relevant = relevant(gt_sim)

    n_rel = is_relevant.sum(dim=1).to(torch.float32)
    _relevant = is_relevant.gather(1, pred_rank[:, :int(n_rel.max())]).to(torch.float32)
    _n_rec_rel = torch.cumprod(_relevant, dim=1).sum(dim=1)
    
    return (_n_rec_rel / n_rel.clamp(min=1e-6)).mean().item()
