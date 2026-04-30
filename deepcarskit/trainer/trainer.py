# @Time   : 2026/02
# @Author : Yong Zheng
# @Notes  : Inherit from recbole.trainer.Trainer

r"""
recbole.trainer.trainer
################################
"""


import torch
import os

from tqdm import tqdm

from deepcarskit.data import LabledDataSortEvalDataLoader
from deepcarskit.evaluator import CARSCollector
from recbole.trainer import Trainer
from recbole.data import FullSortEvalDataLoader
from recbole.utils import EvaluatorType, set_color, get_gpu_usage
from deepcarskit.evaluator import Evaluator

class TrainerWithBestEpoch:
    """
    Wrapper for RecBole Trainer to record best epoch during training.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.best_epoch = None
        self.best_valid_score = None

    def fit(self, train_data, valid_data=None, **kwargs):
        # fit from parent class
        best_score, best_result, _ = self.trainer.fit(train_data, valid_data, **kwargs)

        # iterations to find the best epoch
        best_epoch = None
        for epoch, struct in getattr(self.trainer, "_epoch_structs", {}).items():
            result = self.trainer.evaluator.evaluate(struct)
            score = result.get("valid_score", None)
            if score == self.trainer.best_valid_score:
                best_epoch = epoch
                break

        self.best_epoch = best_epoch
        self.best_valid_score = self.trainer.best_valid_score
        return best_score, best_result, self.best_epoch



class CARSTrainer(Trainer):
    def __init__(self, config, model):
        super(CARSTrainer, self).__init__(config, model)
        self.eval_collector = CARSCollector(config)
        self.evaluator = Evaluator(config)
        self.config = config
        self._epoch_structs = {}  # save struct for each epoch
        self._warned_empty_eval = False

        if self.config['save_per_uc_metrics']:
            self.config['eval_step'] = 1
        self.item_tensor = None

    def _build_empty_eval_result(self):
        result = {}
        metric_names = [metric.lower() for metric in self.config['metrics']]

        for metric in metric_names:
            metric_obj = self.evaluator.metric_class.get(metric)
            if self.config['eval_type'] == EvaluatorType.RANKING and getattr(metric_obj, 'metric_type', None) == EvaluatorType.RANKING:
                for k in self.config['topk']:
                    result[f'{metric}@{k}'] = 0.0
            else:
                result[metric] = 0.0

        if self.config['eval_type'] == EvaluatorType.RANKING and 'precision' in metric_names and 'recall' in metric_names:
            for k in self.config['topk']:
                result[f'f1@{k}'] = 0.0

        return result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=False, model_file=None, show_progress=False, epoch=None):
        if eval_data is None:
            return

        if epoch is None:
            epoch = getattr(self, 'current_epoch', None)

        if load_best_model:
            checkpoint_file = model_file if model_file else self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            self.logger.info(f'Loading model structure and parameters from {checkpoint_file}')

        eval_data_len = len(eval_data) if hasattr(eval_data, '__len__') else None
        if eval_data_len == 0:
            if hasattr(self.eval_collector, 'clear'):
                self.eval_collector.clear()
            if not self._warned_empty_eval:
                self.logger.warning(
                    'Validation/evaluation split contains no positive samples after threshold filtering; '
                    'returning zero metrics for this evaluation pass.'
                )
                self._warned_empty_eval = True
            return self._build_empty_eval_result()

        self.model.eval()

        if isinstance(eval_data, LabledDataSortEvalDataLoader):
            # LabledDataSortEvalDataLoader follows the same batch contract as full-sort evaluation.
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        elif isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        if hasattr(self.eval_collector, 'clear'):
            self.eval_collector.clear()

        iter_data = tqdm(eval_data, total=len(eval_data), ncols=100,
                         desc=set_color("Evaluate", 'pink')) if show_progress else eval_data

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)

        # collect model info
        self.eval_collector.model_collect(self.model)

        # make sure struct has necessary keys
        struct = self.eval_collector.get_data_struct()

        if self.config['save_per_uc_metrics']:
            if epoch is not None:
                self._epoch_structs[epoch] = struct
            else:
                self._epoch_structs[-1] = struct
                self.logger.warning("evaluate called without epoch, saving struct to -1 key")

        result = self.evaluator.evaluate(struct)
        result = {k.lower(): v for k, v in result.items()}
        return result

    def save_best_per_uc(self, best_epoch):
        if best_epoch not in self._epoch_structs:
            self.logger.warning(f"No struct recorded for epoch {best_epoch}")
            return

        struct = self._epoch_structs[best_epoch]
        save_folder = self.config.save_per_uc_folder
        dataset_name = self.config.dataset
        model_name = self.config.model
        fold_num = getattr(self.config, 'current_fold', 0)

        os.makedirs(save_folder, exist_ok=True)
        file_name = f"{dataset_name}_{model_name}_fold{fold_num}.csv"
        file_path = os.path.join(save_folder, file_name)

        self.evaluator.evaluate_per_uc(struct, save_path=file_path)
        self.logger.info(f"Best epoch per-UC metrics saved to {file_path}")

    def fit(self, train_data, valid_data=None, **kwargs):
        """
        Adapted fit function in CARSTrainer
        """
        best_score = None
        best_result = None
        best_epoch = None

        # Use the active valid_metric (set by CARSConfig based on ranking flag)
        # rather than always using ranking_valid_metric when ranking may be False
        metric_key = self.config['valid_metric'].lower()
        bigger = self.valid_metric_bigger  # True for Recall/NDCG/MAP; False for MAE/RMSE

        for epoch_idx in range(self.epochs):
            self.current_epoch = epoch_idx

            super()._train_epoch(
                train_data,
                epoch_idx,
                show_progress=kwargs.get('show_progress', False)
            )

            # ===== validation =====
            if valid_data is not None:
                result = self.evaluate(
                    valid_data,
                    epoch=epoch_idx,
                    show_progress=kwargs.get('show_progress', False)
                )

                if result is None or metric_key not in result:
                    continue

                score = result[metric_key]

                self.logger.info(
                    f"Epoch {epoch_idx} validation {metric_key} = {score:.6f}"
                )

                # ===== select best epoch =====
                is_better = (
                        best_score is None or
                        (score > best_score if bigger else score < best_score)
                )

                if is_better:
                    best_score = score
                    best_result = result
                    best_epoch = epoch_idx

                    # save checkpoint for best epoch
                    self._save_checkpoint(epoch_idx)

        valid_data_len = len(valid_data) if valid_data is not None and hasattr(valid_data, '__len__') else None

        # running for outputing per-uc records (ranking tasks only, since evaluate_per_uc uses rec.topk)
        if self.config['save_per_uc_metrics'] and self.config.get('ranking', False) and best_epoch is not None and valid_data_len != 0:
            self.logger.info(f"Re-evaluating best epoch {best_epoch} for per-UC metrics")

            # load best model
            # This checkpoint is generated internally during training and is trusted.
            # Do NOT use weights_only=True because checkpoint contains CARSConfig.
            checkpoint = torch.load(self.saved_model_file, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))

            # evaluate for each uc pair
            self.evaluate(
                valid_data,
                epoch=best_epoch,
                show_progress=False
            )

            # save
            self.save_best_per_uc(best_epoch)
        elif self.config['save_per_uc_metrics'] and self.config.get('ranking', False) and best_epoch is not None and valid_data_len == 0:
            self.logger.warning(
                'Skipping per-UC metric export because the validation split contains no positive samples.'
            )

        return best_score, best_result, best_epoch

