# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


# UPDATE:
# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : made several changes to adapt it for CARS

"""
deepcarskit.quick_start
########################
"""
import logging
from logging import getLogger
import shutil
import glob
import os

import numpy
import torch
import pickle


# from past.builtins import raw_input

from deepcarskit.config import CARSConfig
from deepcarskit.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from deepcarskit.utils.utils import get_model, get_trainer
from deepcarskit.utils import init_logger, init_seed, set_color
from multiprocessing.dummy import Pool as ThreadPool
from recbole.utils import EvaluatorType


def eval_folds(args_tuple):
    train_data_fold = args_tuple[0]
    valid_data_fold = args_tuple[1]
    config = args_tuple[2]
    logger = args_tuple[3]
    fold = args_tuple[4]

    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data_fold, valid_data_fold))

    # model loading and initialization
    model = get_model(config['model'])(config, train_data_fold.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    name = trainer.saved_model_file
    ind = name.rindex('.')
    lname = list(name)
    lname.insert(ind, '_f'+str(fold))
    trainer.saved_model_file = ''.join(lname)

    # model training
    best_valid_score_fold, best_valid_result_fold = trainer.fit(
        train_data_fold, valid_data_fold, saved=True, show_progress=config['show_progress']
    )
    msghead = 'Fold ' + str(fold) + ' completed: '
    logger.info(set_color(msghead, 'yellow') + f': {best_valid_result_fold}')

    return best_valid_score_fold, best_valid_result_fold


def run(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = CARSConfig(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    log_handler, log_filepath = init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # dataset splitting
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    train_data, valid_data = data_preparation(config, dataset)


    CV = False
    if isinstance(train_data, list):
        CV = True
    n_folds = len(train_data)

    if CV:
        list_train_test = []
        for i in range(n_folds):
            t = (train_data[i], valid_data[i], config, logger, (i+1))
            list_train_test.append(t)

        pool = ThreadPool()
        rsts = pool.map(eval_folds, list_train_test)
        pool.close()
        pool.join()

        best_valid_score = 0
        best_valid_result = {}

        for rst_fold in rsts:
            valid_score_fold = rst_fold[0]
            valid_result_fold = rst_fold[1]

            best_valid_score += valid_score_fold
            if not best_valid_result:
                best_valid_result = valid_result_fold
            else:
                for key in best_valid_result.keys():
                    best_valid_result[key] = best_valid_result[key] + valid_result_fold[key]

        best_valid_score = round(best_valid_score/n_folds, config['metric_decimal_place'])
        for key in best_valid_result:
            best_valid_result[key] = round(best_valid_result[key]/n_folds, config['metric_decimal_place'])
        msghead = 'Data: '+config['dataset']+', Results on '+str(n_folds)+' CV: best valid by '+config['model']
        layers = [str(int) for int in config['mlp_hidden_size']]
        layers = ' '.join(layers)
        logger.info(set_color(msghead, 'yellow') + f': {best_valid_result}'+', lrate: '+str(config['learning_rate'])+', layers: ['+layers+']')
        log_handler.close()
        logger.removeHandler(log_handler)
        logger_name = log_filepath[:-4] + "_" + config['valid_metric'] + " = " + str(best_valid_score) + ".log"
        shutil.move(log_filepath, logger_name)
        update_best_log(config, logger_name, best_valid_result)
    else:
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data))

        # model loading and initialization
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config['show_progress']
        )

        # model evaluation
        # test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

        msghead = 'Data: '+config['dataset']+', best valid by '+config['model']
        logger.info(set_color(msghead, 'yellow') + f': {best_valid_result}')
        # logger.info(set_color('test result', 'yellow') + f': {test_result}')
        log_handler.close()
        logger.removeHandler(log_handler)
        logger_name = log_filepath[:-4] + "_" + config['valid_metric'] + " = " + str(best_valid_score) + ".log"
        shutil.move(log_filepath, logger_name)
        update_best_log(config, logger_name, best_valid_result)

    '''
    # example of predictions by context recommender
    # note, raw value in the original data is expected to be transformed to inner ID
    
    # rawid <--->innderid
    print("innerid: ", dataset._get_innderid_from_rawid("user_id", "1003"))
    print("rawid: ", dataset._get_rawid_from_innerid("user_id", 1))
    
    userid = dataset._get_innderid_from_rawid("user_id","1003")
    itemid = dataset._get_innderid_from_rawid("item_id","tt0120912")
    timeid = dataset._get_innderid_from_rawid("time","Weekday")
    locid = dataset._get_innderid_from_rawid("location","Cinema")
    cmpid = dataset._get_innderid_from_rawid("companion","Alone")

    user = torch.tensor([userid])
    item = torch.tensor([itemid])
    contexts = []
    contexts.append(torch.tensor([timeid]))
    contexts.append(torch.tensor([locid]))
    contexts.append(torch.tensor([cmpid]))
    print(userid, ', ', itemid, ', ', timeid, ', ', locid, ', ', cmpid)
    print("prediction: ",model.forward(user, item, contexts))
    exit()
    '''

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        # 'test_result': test_result
    }

def update_best_log(config, newlog, best_valid_result):
    dataset = config['dataset']
    # compare which log file is better
    ranking = False
    if config['eval_type'] == EvaluatorType.RANKING:
        ranking = True
        metric = config['ranking_valid_metric']
    else:
        metric = config['err_valid_metric']

    metric_value = best_valid_result[metric.lower()]

    end = newlog.rindex('.')
    s1 = newlog.index('-')
    s2 = newlog.index('-', s1 + 1, end)
    model = newlog[s1 + 1:s2]

    match = [dataset, model, metric]


    folder_best = './log/best/'
    existing_logs = glob.glob(folder_best+'/*.log')

    found = False
    oldlog = None
    for file in existing_logs:
        if all(x in file for x in match):
            oldlog = file
            found = True
            break

    newlog_filename = newlog[newlog.rindex('/')+1:]

    if not found:
        shutil.copyfile(newlog, folder_best+newlog_filename)
    else:
        newvalue = metric_value
        oldvalue = float(oldlog[oldlog.rindex('=') + 1: oldlog.rindex('.')])

        if ranking:
            if newvalue > oldvalue:
                shutil.copyfile(newlog, folder_best+newlog_filename)
                os.remove(oldlog)
                impro = (newvalue - oldvalue) / oldvalue
                print('Better results! improvement: {:.2%}'.format(impro) + ', best log saved in ' + folder_best + newlog_filename)
        else:
            if newvalue < oldvalue:
                shutil.copyfile(newlog, folder_best+newlog_filename)
                os.remove(oldlog)
                impro = (oldvalue - newvalue) / oldvalue
                print('Better results! improvement: {:.2%}'.format(impro) + ', best log saved in ' + folder_best + newlog_filename)
    return



def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = CARSConfig(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    # test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        # 'test_result': test_result
    }


def load_data_and_model(model_file, dataset_file=None, dataloader_file=None):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.
        dataset_file (str, optional): The path of filtered dataset. Defaults to ``None``.
        dataloader_file (str, optional): The path of split dataloaders. Defaults to ``None``.

    Note:
        The :attr:`dataset` will be loaded or created according to the following strategy:
        If :attr:`dataset_file` is not ``None``, the :attr:`dataset` will be loaded from :attr:`dataset_file`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is ``None``,
        the :attr:`dataset` will be created according to :attr:`config`.
        If :attr:`dataset_file` is ``None`` and :attr:`dataloader_file` is not ``None``,
        the :attr:`dataset` will neither be loaded or created.

        The :attr:`dataloader` will be loaded or created according to the following strategy:
        If :attr:`dataloader_file` is not ``None``, the :attr:`dataloader` will be loaded from :attr:`dataloader_file`.
        If :attr:`dataloader_file` is ``None``, the :attr:`dataloader` will be created according to :attr:`config`.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_logger(config)

    dataset = None
    if dataset_file:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

    if dataloader_file:
        train_data, valid_data, test_data = load_split_dataloaders(dataloader_file)
    else:
        if dataset is None:
            dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
