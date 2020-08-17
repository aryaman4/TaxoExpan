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
import sys
from test_fast import encode_graph, rearrange
import itertools
from data_loader.dataset import MAGDataset

class TaxoClean(object):
    def __init__(self, config_path="./TaxoExpan/config_files/config.mag.json"):
        self.config = ConfigParser(default_vals={'config_path': config_path})
        self.ranking_dict = {}
        self.node_cov = {}
       
    def run_trainer(self):
        trainer = Trainer(self.model, self.loss, self.metrics, self.pre_metric, self.optimizer,
                            config=self.config,
                            data_loader=self.train_data_loader,
                            valid_data_loader=self.validation_data_loader,
                            lr_scheduler=self.lr_scheduler)

        trainer.train()

    def initialize_data(self):
        binary_dataset = MAGDataset(name="computer_science", path="./TaxoExpan/data/MAG-CS/", raw=True)
        self.train_data_loader = self.config.initialize('train_data_loader', module_data, "train")
        self.validation_data_loader = self.config.initialize('validation_data_loader', module_data, "validation")
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
        test_data_path = self.config['test_data_loader']['args']['data_path']
        self.test_data_loader = module_data.MaskedGraphDataLoader(
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
        self.full_size = len(self.train_data_loader.dataset.node_list) + len(self.validation_data_loader.dataset.node_list) + len(self.test_data_loader.dataset.node_list)

    def run_ranking(self):
        logger = self.config.get_logger('test')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # metric_fns = [getattr(module_metric, met) for met in self.config['metrics']]
        test_dataset = self.test_data_loader.dataset
        kv = test_dataset.kv
        vocab = test_dataset.node_list
        indice2word = test_dataset.vocab
        node2parents = test_dataset.node2parents
        candidate_positions = sorted(list(test_dataset.all_positions))
        logger.info("Number of queries: {}".format(len(vocab)))
        anchor2subgraph = {}
        for anchor in tqdm(candidate_positions):
            anchor2subgraph[anchor] = test_dataset._get_subgraph(-1, anchor, 0)
        
        # obtain graph representation
        bg = dgl.batch([v for k,v in anchor2subgraph.items()])
        h = bg.ndata.pop('x').to(device)
        candidate_position_idx = bg.ndata['_id'][bg.ndata['pos']==1].tolist()
        n_position = len(candidate_position_idx)
        pos = bg.ndata['pos'].to(device)
        with torch.no_grad():
            hg = encode_graph(self.model, bg, h, pos)

        # start per query prediction
        # total_metrics = torch.zeros(len(metric_fns))
        with torch.no_grad():
            for query in vocab:
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                expanded_nf = nf.expand(n_position, -1)
                energy_scores = self.model.match(hg, expanded_nf)
                predicted_scores = energy_scores.cpu().squeeze_().tolist()
                if self.config['loss'].startswith("info_nce"):
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])]
                else:
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])]
                rank_dict = {}
                for predict_index, parent_index in enumerate(predict_parent_idx_list):
                    rank_dict[parent_index] = predict_index
                true_parent_indices = node2parents[query]
                true_parent_rank_dict = {}
                for tp_index in true_parent_indices:
                    true_parent_rank_dict[indice2word[tp_index]] = rank_dict[tp_index]
                print(indice2word[query])
                print(true_parent_rank_dict)
                q = indice2word[query]
                if q not in self.ranking_dict:
                    self.ranking_dict[q] = []
                for rank in true_parent_rank_dict.values():
                    self.ranking_dict[q].append(rank)
    def run_full_routine(self):
        i = 0
        while True:
            print(i)
            i+=1
            self.initialize_data()
            test_dataset = self.test_data_loader.dataset
            vocab = test_dataset.node_list
            indice2word = test_dataset.vocab
            for node in vocab:
                self.node_cov[indice2word[node]] = 1
            print(len(self.node_cov.keys()))
            print(self.full_size)
            self.run_trainer()
            self.run_ranking()
            if len(self.node_cov.keys()) == self.full_size:
                break
            


if __name__ == '__main__':
    tc = TaxoClean()
    f = open("rank_results.txt", "w+")
    tc.run_full_routine()
    for k, v in tc.ranking_dict.items():
        f.write(str(k))
        for rank in v:
            f.write(" ")
            f.write(str(rank))
        f.write("\n")
