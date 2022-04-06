# @Time   : 2021/12
# @Author : Yong Zheng
# @Notes  : added F1 metrics, if precision and recall defined in user requests

"""
deepcarskit.evaluator.evaluator
#####################################
"""
import numpy as np
from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            dict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = {}
        topk = []
        metric_f1 = False
        if self.config['ranking']:
            topk = self.config['topk']
            if 'precision' in self.metrics and 'recall' in self.metrics:
                metric_f1 = True

        for metric in self.metrics:
            # dataobject has two keys: rec.score, data.label
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)

            # adding F1 metric, if precision and recall were calculated
            if metric_f1:
                k = topk[0]
                keys = result_dict.keys()
                key1 = 'precision@'+str(k)
                key2 = 'recall@'+str(k)
                key = 'f1@'+str(k)
                if key1 in keys and key2 in keys and key not in keys:
                    metric = {}
                    for k in topk:
                        key1 = 'precision@'+str(k)
                        key2 = 'recall@'+str(k)
                        key = 'f1@'+str(k)
                        precision = result_dict[key1]
                        recall = result_dict[key2]
                        if (precision + recall) == 0:
                            f1 = 0
                        else:
                            f1 = round(2*precision*recall/(precision + recall), self.config['metric_decimal_place'])
                        metric[key] = f1
                    result_dict.update(metric)
        return result_dict
