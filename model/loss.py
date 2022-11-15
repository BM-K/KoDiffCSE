import torch
import logging
import numpy as np
import torch.nn as nn
from model.utils import Metric
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

logger = logging.getLogger(__name__)


class Loss():

    def __init__(self, args):
        self.args = args
        self.cos = nn.CosineSimilarity(dim=-1)
        self.metric = Metric(args)

    def train_loss_fct(self, config, inputs, z1, z2, diff_outputs):
         
        cosine_similarity = self.cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.args.temperature
        
        labels = torch.arange(cosine_similarity.size(0)).long().to(self.args.device)
        loss = config['criterion'](cosine_similarity, labels)

        diff_loss = 0
        type_labels = diff_outputs[1:-1]
        for type_pred_label, label in zip(diff_outputs[0], type_labels):
            diff_loss += config['criterion'](type_pred_label.view(-1, 2), label.view(-1)) * self.args.lambda_weight
            
        return loss + diff_loss

    def evaluation_during_training(self, embeddings1, embeddings2, labels, indicator):

        embeddings1 = embeddings1.cpu().numpy()
        embeddings2 = embeddings2.cpu().numpy()
        labels = labels['value'].cpu().numpy().flatten()

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
        
        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        score = {'eval_pearson_cosine': eval_pearson_cosine,
                 'eval_spearman_cosine': eval_spearman_cosine,
                 'eval_pearson_manhattan': eval_pearson_manhattan,
                 'eval_spearman_manhattan': eval_spearman_manhattan,
                 'eval_pearson_euclidean': eval_pearson_euclidean,
                 'eval_spearman_euclidean': eval_spearman_euclidean,
                 'eval_pearson_dot': eval_pearson_dot,
                 'eval_spearman_dot': eval_spearman_dot}

        self.metric.update_indicator(indicator, score)
        
        return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
