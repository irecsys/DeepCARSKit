# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : added LabledDataSortEvalDataLoader for context-aware ranking evaluations

"""
deepcarskit.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType
from collections import defaultdict
from logging import getLogger

class FullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False, used_ids=None):
        super().__init__(config, dataset, sampler, shuffle=shuffle)


class LabledDataSortEvalDataLoader(FullSortEvalDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

        used_item = all items that users have interacted in the training and evaluation set.
        positve_item = all items that users have interacted in the evaluation set.
        history_item = all items that users have interacted in the training set.
    """

    def __init__(self, config, dataset, sampler, shuffle=False, used_ids=None):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config['MODEL_TYPE'] == ModelType.SEQUENTIAL

        self.user_id = config['USER_ID_FIELD']
        self.item_id = config['ITEM_ID_FIELD']
        self.uc_id = config['USER_CONTEXT_FIELD']
        self.LABEL = config['LABEL_FIELD']

        if not self.is_sequential:
            multidict_uc_items = self._get_multidict(dataset) # uc and rated items

            '''    
                uc_positve_item = all items that uc have rated and must be positive in the evaluation set.
                uc_history_item = all items that uc have rated in the training set.
            '''
            self.ucid_list = multidict_uc_items.keys()
            self.uc_num = max(self.ucid_list)+1
            self.ucid2items_num = np.zeros(self.uc_num, dtype=np.int64)
            self.ucid2positive_item = np.array([None] * self.uc_num)
            self.ucid2history_item = np.array([None] * self.uc_num)
            self.ucid_condidates={}

            # rated items (positive AND negative) for each uc in the training set
            ucid2used_item = used_ids

            for ucid in self.ucid_list:

                uc_positve_itemlist = set(multidict_uc_items[ucid])
                self.ucid2positive_item[ucid] = torch.tensor(list(uc_positve_itemlist), dtype=torch.int64)

                self.ucid2items_num[ucid] = len(uc_positve_itemlist)

                uc_history_itemlist = ucid2used_item[ucid]

                self.ucid2history_item[ucid] = torch.tensor(list(uc_history_itemlist), dtype=torch.int64)

            # get uid and context information from uc innerid
            context_fields = dataset._get_context_fields()
            uid_list = []
            dict_context = {}
            for context in context_fields:
                dict_context[context]=[]

            for ucid in self.ucid_list:
                uid = dataset._get_uid_from_usercontexts(ucid)
                uid_list.append(uid)
                tuple_context = dataset._get_context_tuple_from_usercontexts(ucid)
                for i in range(0,len(context_fields)):
                    context = context_fields[i]
                    dict_context[context].append(tuple_context[i])

            self.ucid_list = torch.tensor(list(self.ucid_list), dtype=torch.int64)
            uid_list = torch.tensor(list(uid_list), dtype=torch.int64)
            # add uc data into data for predictions
            self.uc_df = dataset.join(Interaction({self.uid_field: uid_list, self.uc_id: self.ucid_list}))
            for context in dict_context.keys():
                new_inter = dataset.join(Interaction({context: torch.tensor(list(dict_context[context]), dtype=torch.int64)}))
                self.uc_df.update(new_inter)


        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = self.step = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _get_multidict(self, dataset):
        matrix_uc_item = dataset._create_sparse_matrix(dataset.inter_feat, self.uc_id, self.item_id, 'coo',
                                                       self.LABEL)
        # multidict_u_uc = defaultdict(list)
        # key = userid, value = Dict (key = uc, value = decending ranked items with ratings)
        # will get a list of Dict, given a userid
        multidict_uc_items = defaultdict(list)
        multidict_uc_items_positives = defaultdict(list)


        rows, cols = matrix_uc_item.shape
        for uc_id in range(1, rows):
            # Index = 0 => [PAD]
            # uc_id == inner id for user_context
            uc_items = matrix_uc_item.getrow(uc_id)  # csr_matrix
            items = uc_items.indices  # a list of items
            rates = uc_items.data  # a list of ratings
            num_rates = len(rates)

            if num_rates == 0:
                continue

            dict_item_rating = {}

            for i in range(0, num_rates):
                key = items[i]
                value = rates[i]
                dict_item_rating[key] = value
            # sort items based on ratings
            dict_item_rating_decending = sorted(dict_item_rating.items(), key=lambda x: x[1], reverse=True)
            # add these items into dict which uses uc as key
            for items in dict_item_rating_decending:
                multidict_uc_items[uc_id].append(items[0])
        return multidict_uc_items

    @property
    def pr_end(self):
        if not self.is_sequential:
            return len(self.ucid_list)
        else:
            return len(self.dataset)

    def _next_batch_data(self):
        if not self.is_sequential:
            uc_df = self.uc_df[self.pr:self.pr + self.step]
            ucid_list = list(uc_df[self.uc_id])

            history_item = self.ucid2history_item[ucid_list]
            positive_item = self.ucid2positive_item[ucid_list]

            history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)])
            positive_i = torch.cat(list(positive_item))

            self.pr += self.step
            return uc_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self.dataset[self.pr:self.pr + self.step]
            inter_num = len(interaction)
            positive_u = torch.arange(inter_num)
            positive_i = interaction[self.iid_field]

            self.pr += self.step
            return interaction, None, positive_u, positive_i



