
# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : Inherit from recbole.model.layers.FMFirstOrderLinear



"""
deepcarskit.model.layers
#############################
Common Layers in recommender system
"""

from recbole.model.layers import FMFirstOrderLinear
from recbole.model.layers import FMEmbedding

import numpy as np
import torch
import torch.nn as nn

from recbole.utils import FeatureType, FeatureSource



class FMFirstOrderLinear(FMFirstOrderLinear):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """

    def __init__(self, config, dataset, output_dim=1):

        super(FMFirstOrderLinear, self).__init__(config, dataset, output_dim)
        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        if config['ranking']:
            self.LABEL = config['LABEL_FIELD']
        else:
            self.LABEL = config['RATING_FIELD']
        self.device = config['device']
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        for field_name in self.field_names:
            if field_name == config['RATING_FIELD'] or field_name == config['LABEL_FIELD']:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims, self.token_field_offsets, output_dim)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(np.sum(self.float_field_dims, dtype=np.int32), output_dim)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, output_dim))

        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)