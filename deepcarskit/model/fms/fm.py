# -*- coding: utf-8 -*-
# @Time   : 2020/7/8 10:09
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : fms.py

# UPDATE:
# @Time   : 2020/8/13,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

# UPDATE:
# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : made changes to adapt it for CARS

r"""
FM
################################################
References
-----
Steffen Rendle et al. "Factorization Machines." in ICDM 2010.

Notes
-----
context variables are treated as individual dimensions
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_

from deepcarskit.model.context_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine
from recbole.utils import EvaluatorType


class FM(ContextRecommender):
    """Factorization Machine considers the second-order interaction with features to predict the final score.

    """

    def __init__(self, config, dataset):

        super(FM, self).__init__(config, dataset)

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.config = config

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.actfun = nn.Sigmoid()
            self.loss = nn.BCELoss()
            self.LABEL = self.config['LABEL_FIELD']
        else:
            self.actfun = nn.LeakyReLU()
            self.loss = nn.MSELoss()
            self.LABEL = self.config['RATING_FIELD']

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        fm_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        y = self.actfun(self.first_order_linear(interaction) + self.fm(fm_all_embeddings))
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
