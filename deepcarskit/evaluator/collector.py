# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : Inherit from recbole.evaluator.Collector

"""
recbole.evaluator.collector
################################################
"""

from recbole.evaluator.register import Register
from recbole.evaluator import Collector, DataStruct
import torch


class CARSCollector(Collector):
    """The collector is used to collect the resource for evaluator.
        As the evaluation metrics are various, the needed resource not only contain the recommended result
        but also other resource from data and model. They all can be collected by the collector during the training
        and evaluation process.

        This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = ('full' in config['eval_args']['mode'])
        self.topk = self.config['topk']
        self.device = self.config['device']

    def eval_batch_collect(
        self, scores_tensor: torch.Tensor, interaction, positive_u: torch.Tensor, positive_i: torch.Tensor
    ):
        """ Collect the evaluation resource from batched eval data and batched model output.
            Args:
                scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                interaction(Interaction): batched eval data.
                positive_u(Torch.Tensor): the row index of positive items for each user.
                positive_i(Torch.Tensor): the positive item id for each user.
        """
        if self.register.need('rec.items'):

            # get topk
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            self.data_struct.update_tensor('rec.items', topk_idx)

        if self.register.need('rec.topk'):

            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.topk', result)

        if self.register.need('rec.meanrank'):

            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.meanrank', result)

        if self.register.need('rec.score'):

            self.data_struct.update_tensor('rec.score', scores_tensor)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            if self.config['ranking']:
                self.data_struct.update_tensor('data.label', interaction[self.label_field].to(self.device))
            else:
                self.data_struct.update_tensor('data.label', interaction[self.config['RATING_FIELD']].to(self.device))
