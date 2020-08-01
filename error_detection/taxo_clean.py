import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from functools import partial
import time
from tqdm import tqdm
import dgl
from test_fast import encode_graph, rearrange
import itertools

class TaxoClean(object):
    def __init__(self, config_path="../config_files/config.mag.json", case_study=True):
        self.config = ConfigParser(default_vals={'config_path': config_path})
        self.train_data_loader = self.config.initialize('train_data_loader', module_data, "train")
        self.validation_data_loader = self.config.initialize('validation_data_loader', module_data, "validation")
        self.case_study = case_study
        # build model architecture, then print to console
        self.model = self.config.initialize('arch', module_arch)

        # get function handles of loss and metrics
        self.loss = getattr(module_loss, self.config['loss'])
        self.metrics = [getattr(module_metric, met) for met in self.config['metrics']]
        if self.config['loss'].startswith("info_nce"):
            self.pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
        else:
            self.pre_metric = partial(module_metric.obtain_ranks, mode=0)

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.config.initialize('optimizer', torch.optim, self.trainable_params)
        self.lr_scheduler = self.config.initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
    
    def run_trainer(self):
        trainer = Trainer(self.model, self.loss, self.metrics, self.pre_metric, self.optimizer,
                            config=self.config,
                            data_loader=self.train_data_loader,
                            valid_data_loader=self.validation_data_loader,
                            lr_scheduler=self.lr_scheduler)

        trainer.train()

    def initialize_data(self):
        self.train_data_loader = self.config.initialize('train_data_loader', module_data, "train")
        self.validation_data_loader = self.config.initialize('validation_data_loader', module_data, "validation")
    
    def run_ranking(self):
        need_case_study = self.case_study
        logger = self.config.get_logger('test')
        test_data_path = self.config['test_data_loader']['args']['data_path']
        test_data_loader = module_data.MaskedGraphDataLoader(
            mode="test", 
            data_path=test_data_path,
            sampling_mode=0,
            batch_size=1, 
            expand_factor=self.config['test_data_loader']['args']['expand_factor'], 
            shuffle=True, 
            num_workers=8, 
            batch_type="large_batch", 
            cache_refresh_time=self.config['test_data_loader']['args']['cache_refresh_time'],
            normalize_embed=self.config['test_data_loader']['args']['normalize_embed']
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metric_fns = [getattr(module_metric, met) for met in self.config['metrics']]
        test_dataset = test_data_loader.dataset
        kv = test_dataset.kv
        vocab = test_dataset.node_list
        if need_case_study:
            indice2word = test_dataset.vocab
        node2parents = test_dataset.node2parents
        candidate_positions = sorted(list(test_dataset.all_positions))
        logger.info("Number of queries: {}".format(len(vocab)))
        anchor2subgraph = {}
        for anchor in tqdm(candidate_positions):
            anchor2subgraph[anchor] = test_dataset._get_subgraph(-1, anchor, 0)
        
        if self.config['test_data_loader']['args']['batch_size'] == 1:  # small dataset with only one batch
            # obtain graph representation
            bg = dgl.batch([v for k,v in anchor2subgraph.items()])
            h = bg.ndata.pop('x').to(device)
            candidate_position_idx = bg.ndata['_id'][bg.ndata['pos']==1].tolist()
            n_position = len(candidate_position_idx)
            pos = bg.ndata['pos'].to(device)
            with torch.no_grad():
                hg = encode_graph(self.model, bg, h, pos)

            # start per query prediction
            total_metrics = torch.zeros(len(metric_fns))
            if need_case_study:
                all_cases = []
                all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])
            with torch.no_grad():
                for i, query in tqdm(enumerate(vocab)):
                    if need_case_study:
                        cur_case = [indice2word[query]]
                        true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                        cur_case.append(true_parents)
                    nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                    expanded_nf = nf.expand(n_position, -1)
                    energy_scores = self.model.match(hg, expanded_nf)
                    if need_case_study:  # select top-5 predicted parents
                        predicted_scores = energy_scores.cpu().squeeze_().tolist()
                        if self.config['loss'].startswith("info_nce"):
                            predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                        else:
                            predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                        predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                        cur_case.append(predict_parents)
                    energy_scores, labels = rearrange(energy_scores, candidate_position_idx, node2parents[query])
                    all_ranks = self.pre_metric(energy_scores, labels)
                    for j, metric in enumerate(metric_fns):
                        tmp = metric(all_ranks)
                        total_metrics[j] += tmp
                        if need_case_study:
                            cur_case.append(str(tmp))
                    if need_case_study:
                        all_cases.append(cur_case)
            
            # save case study results to file
            if need_case_study:
                with open("case_study.tsv", "w") as fout:
                    for ele in all_cases:
                        fout.write("\t".join(ele))
                        fout.write("\n")

        else:  # large dataset with many batches
            # obtain graph representation
            logger.info('Large batch mode with batch_size = {}'.format(self.config['test_data_loader']['args']['batch_size']))
            batched_hg = []  # save the CPU graph representation
            batched_positions = []
            bg = []
            positions = []
            with torch.no_grad():
                for i, (anchor, egonet) in tqdm(enumerate(anchor2subgraph.items()), desc="Generating graph encoding ..."):
                    positions.append(anchor)
                    bg.append(egonet)
                    if (i+1) % self.config['test_data_loader']['args']['batch_size'] == 0:
                        bg = dgl.batch(bg)
                        h = bg.ndata.pop('x').to(device)
                        pos = bg.ndata['pos'].to(device)
                        hg = encode_graph(self.model, bg, h, pos)
                        assert hg.shape[0] == len(positions), "mismatch between hg.shape[0]: {} and len(positions): {}".format(hg.shape[0],len(positions))
                        batched_hg.append(hg.cpu())
                        batched_positions.append(positions)
                        bg = []
                        positions = []
                        del h
                if len(bg) != 0:
                    bg = dgl.batch(bg)
                    h = bg.ndata.pop('x').to(device)
                    pos = bg.ndata['pos'].to(device)
                    hg = encode_graph(self.model, bg, h, pos)
                    assert hg.shape[0] == len(positions), "mismatch between hg.shape[0]: {} and len(positions): {}".format(hg.shape[0], len(positions))
                    batched_hg.append(hg.cpu())
                    batched_positions.append(positions)
                    del h
        
            # start per query prediction
            total_metrics = torch.zeros(len(metric_fns))
            if need_case_study:
                all_cases = []
                all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])
            candidate_position_idx = list(itertools.chain(*batched_positions))
            batched_hg = [hg.to(device) for hg in batched_hg]
            with torch.no_grad():
                for i, query in tqdm(enumerate(vocab)):
                    if need_case_study:
                        cur_case = [indice2word[query]]
                        true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                        cur_case.append(true_parents)
                    nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                    batched_energy_scores = []
                    for hg, positions in zip(batched_hg, batched_positions):
                        n_position = len(positions)
                        expanded_nf = nf.expand(n_position, -1)
                        energy_scores = self.model.match(hg, expanded_nf)  # a tensor of size (n_position, 1)
                        batched_energy_scores.append(energy_scores)
                    batched_energy_scores = torch.cat(batched_energy_scores)
                    if need_case_study:
                        predicted_scores = batched_energy_scores.cpu().squeeze_().tolist()
                        if self.config['loss'].startswith("info_nce"):
                            predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                        else:
                            predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                        predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                        cur_case.append(predict_parents)
                    batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_position_idx, node2parents[query])
                    all_ranks = self.pre_metric(batched_energy_scores, labels)
                    for j, metric in enumerate(metric_fns):
                        tmp = metric(all_ranks)
                        total_metrics[j] += tmp
                        if need_case_study:
                            cur_case.append(str(tmp))
                    if need_case_study:
                        all_cases.append(cur_case)

            # save case study results to file
            if need_case_study:
                with open("case_study.tsv", "w") as fout:
                    for ele in all_cases:
                        fout.write("\t".join(ele))
                        fout.write("\n")

        n_samples = test_data_loader.n_samples
        log = {}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        log.update({
            "test_topk": test_data_loader.dataset.test_topk
        })
        logger.info(log)