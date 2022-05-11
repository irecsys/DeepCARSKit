# @Time   : 2021/12
# @Author : Yong Zheng

r"""
NeuCMF0i
################################################
References
-----
Yong Zheng, Gonzalo Florez Arias. "A Family of Neural Contextual Matrix Factorization Models for Context-Aware Recommendations", ACM UMAP, 2022

Notes
-----
1). NeuCMF0i has 2 towers (MLP and MF), and it fuses contexts into MLP tower only.

2). NeuCMF0i creates embedding for each individual context conditions.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from deepcarskit.model.context_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType, EvaluatorType


class NeuCMF0i(ContextRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuCMF0i, self).__init__(config, dataset)

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.context_dimensions_mlp_embedding = []
        for i in range(0, self.n_contexts_dim):
            self.context_dimensions_mlp_embedding.append(nn.Embedding(self.n_contexts_conditions[i], self.mlp_embedding_size).to(self.device))

        # mlp layers = user, item, context_situation
        self.mlp_layers = MLPLayers([(2 + self.n_contexts_dim) * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item, context_situation_list):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        context_situation_e = None
        for i in range(0, self.n_contexts_dim):
            condition = context_situation_list[i]
            embd = self.context_dimensions_mlp_embedding[i](condition)
            if context_situation_e is None:
                context_situation_e = embd
            else:
                context_situation_e = torch.cat((context_situation_e, embd), -1)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e, context_situation_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.actfun(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.actfun(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.actfun(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        context_situation_list = self.getContextSituationList(interaction, self.CONTEXTS)
        label = interaction[self.LABEL]

        output = self.forward(user, item, context_situation_list)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        context_situation_list = self.getContextSituationList(interaction, self.CONTEXTS)
        return self.forward(user, item, context_situation_list)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)