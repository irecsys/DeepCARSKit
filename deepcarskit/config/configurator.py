# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : Inherit from recbole.config

"""
deepcarskit.config.configurator
################################
"""

from deepcarskit.utils.utils import get_model
from recbole.config import Config


class CARSConfig(Config):

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        super(CARSConfig, self).__init__(model, dataset, config_file_list, config_dict)

    def _get_model_and_dataset(self, model, dataset):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            # if model is a class object
            final_model_class = model
            final_model = model.__name__
        else:
            # if model is a name in string format
            final_model = model
            final_model_class = get_model(final_model)  # need to get class object

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        # turn on corresponding metrics according to the recommendation task
        if final_config_dict['ranking']:
            final_config_dict['metrics'] = final_config_dict['ranking_metrics']
            final_config_dict['valid_metric'] = final_config_dict['ranking_valid_metric']
        else:
            final_config_dict['metrics'] = final_config_dict['err_metrics']
            final_config_dict['valid_metric'] = final_config_dict['err_valid_metric']
        return final_config_dict
