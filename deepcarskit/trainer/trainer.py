# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : Inherit from recbole.trainer.Trainer

r"""
recbole.trainer.trainer
################################
"""


import numpy as np
import torch

from tqdm import tqdm

from deepcarskit.data import LabledDataSortEvalDataLoader
from deepcarskit.evaluator import CARSCollector
from recbole.trainer import Trainer
from recbole.data import FullSortEvalDataLoader
from recbole.utils import EvaluatorType, set_color, get_gpu_usage
from deepcarskit.evaluator import Evaluator

class CARSTrainer(Trainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(CARSTrainer, self).__init__(config, model)
        self.eval_collector = CARSCollector(config)
        self.evaluator = Evaluator(config)

    def _labled_data_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i


    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        elif isinstance(eval_data, LabledDataSortEvalDataLoader):
            eval_func = self._labled_data_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result


