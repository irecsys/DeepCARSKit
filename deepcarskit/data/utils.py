# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn

# UPDATE:
# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : made several changes to adapt it for CARS


"""
deepcarskit.data.utils
########################
"""

import copy
import importlib
import os
import pickle

from deepcarskit.data.dataloader import *
from recbole.data.dataloader import TrainDataLoader, NegSampleEvalDataLoader, KnowledgeBasedDataLoader, UserDataLoader
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils import EvaluatorType
from logging import getLogger


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataset: Constructed dataset.
    """
    # David Wang: import the model dynamically
    dataset_module = importlib.import_module('deepcarskit.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        """ David Wang:
        if a data set is name after <model_name>Dataset in custom data set model, return the data set class object
        """
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        model_type = config['MODEL_TYPE']
        if model_type == ModelType.SEQUENTIAL:
            from .dataset import SequentialDataset
            return SequentialDataset(config)
        elif model_type == ModelType.KNOWLEDGE:
            from .dataset import KnowledgeBasedDataset
            return KnowledgeBasedDataset(config)
        elif model_type == ModelType.DECISIONTREE:
            from .dataset import DecisionTreeDataset
            return DecisionTreeDataset(config)
        else:
            from .dataset import Dataset
            return Dataset(config)


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color('Saved split dataloaders', 'blue') + f': {file_path}')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)


def load_split_dataloaders(saved_dataloaders_file):
    """Load split dataloaders.

    Args:
        saved_dataloaders_file (str): The path of split dataloaders.

    Returns:
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    with open(saved_dataloaders_file, 'rb') as f:
        dataloaders = pickle.load(f)
    return dataloaders


def data_preparation(config, dataset, save=False):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config['MODEL_TYPE']
    # David Wang: make a copy since dataset.build() will modify the .inter_feat attribute to Interaction object
    dataset = copy.copy(dataset)
    # David Wang: read data file and create 3 pandas DateFrame data sets



    CV = True
    built_datasets = dataset.build()
    if isinstance(built_datasets, list):
        CV = False
    logger = getLogger()

    # dict
    # key = number of fold
    # value = [train, valid set]

    if CV:
        train = []
        valid = []
        for fold in built_datasets:
            train_dataset, valid_dataset = built_datasets[fold]
            train_sampler, valid_sampler = create_samplers(config, dataset, built_datasets[fold])
            used_ids = get_used_ids(config, dataset=train_dataset)

            if model_type != ModelType.KNOWLEDGE:
                train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
            else:
                kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
                train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

            if config['ranking']:
                valid_data_loader = get_dataloader(config, 'evaluation')
                valid_data = valid_data_loader(config, valid_dataset, valid_sampler, shuffle=False, used_ids=used_ids)
            else:
                valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)

            logger.info(
                set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
                set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
                set_color(f'[{config["neg_sampling"]}]', 'yellow')
            )
            logger.info(
                set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
                set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
                set_color(f'[{config["eval_args"]}]', 'yellow')
            )
            train.append(train_data)
            valid.append(valid_data)
            # if save:
                # save_split_dataloaders(config, dataloaders=(train_data, valid_data))

        return train, valid
    else:
        train_dataset, valid_dataset = built_datasets
        train_sampler, valid_sampler = create_samplers(config, dataset, built_datasets)
        used_ids = get_used_ids(config, dataset=train_dataset)

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        else:
            kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

        if config['ranking']:
            valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False, used_ids=used_ids)
        else:
            valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)

        logger.info(
            set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
            set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
            set_color(f'[{config["neg_sampling"]}]', 'yellow')
        )
        logger.info(
            set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
            set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
            set_color(f'[{config["eval_args"]}]', 'yellow')
        )
        if save:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data))

        return train_data, valid_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        'MacridVAE': _get_AE_dataloader,
        'CDAE': _get_AE_dataloader,
        'ENMF': _get_AE_dataloader,
        'RaCT': _get_AE_dataloader,
        'RecVAE': _get_AE_dataloader,
    }

    if config['model'] in register_table:
        return register_table[config['model']](config, phase)

    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            if config['eval_type'] == EvaluatorType.RANKING:
                return LabledDataSortEvalDataLoader
            else:
                return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader

def get_used_ids(config, dataset):
    """
    Returns:
        dict: Used item_ids is the same as positive item_ids.
        Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
    """
    used_item_id = None
    uc_num = dataset.user_context_num
    iid_field = dataset.iid_field
    ucid_field = dataset.ucid_field
    last = [set() for _ in range(uc_num)]
    cur = np.array([set(s) for s in last])
    for ucid, iid in zip(dataset.inter_feat[ucid_field].numpy(), dataset.inter_feat[iid_field].numpy()):
        cur[ucid].add(iid)
    last = used_item_id = cur

    for used_item_set in used_item_id:
        if len(used_item_set) + 1 == dataset.item_num:  # [pad] is a item.
            raise ValueError(
                'Some users have interacted with all items, '
                'which we can not sample negative items for them. '
                'Please set `user_inter_num_interval` to filter those users.'
            )
    return used_item_id

def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return UserDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']

    sampler = None
    train_sampler, valid_sampler = None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')

    return train_sampler, valid_sampler

'''
def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler
'''