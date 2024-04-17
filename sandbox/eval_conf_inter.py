#!/usr/bin/env python
"""
Ferrer, L. and Riera, P. 
Confidence Intervals for evaluation in machine learning [Computer software]. 
https://github.com/luferrer/ConfidenceIntervals

MIT License

Copyright (c) 2023 Luciana Ferrer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


def metric_wrapper(labels, samples, samples2, metric, indices=None):
    """ Call the metric depending on which arguments are given and which are None. 
    labels and samples2 may be None, samples and metric should always be given."""

    if samples is None:
        raise Exception("The samples argument cannot be None")

    if indices is None:
        l, s, s2 = labels, samples, samples2
    else:
        l = labels[indices] if labels is not None else None
        s = samples[indices]
        s2 = samples2[indices] if samples2 is not None else None

    if labels is not None:
        if samples2 is not None:
            return metric(l, s, s2)
        else:
            return metric(l, s)
    else:
        if samples2 is not None:
            return metric(s, s2)
        else:
            return metric(s)


def evaluate_with_conf_int(samples, metric, labels=None, conditions=None, 
                           num_bootstraps=1000, alpha=5, samples2=None):
    """ Evaluate the metric on the provided data and then run bootstrapping to get a confidence interval.
        
        - samples: array of decisions/scores/losses for each sample needed to compute the metric.
                
        - metric: function that computes the metric given a set of samples. The function will be 
          called internally as metric([labels], samples, [samples2]), where the two arguments in 
          brackets are optional (if they are None, they are excluded from the call). 
        
        - labels: array of labels or any per-sample value needed to compute the metric. May be None
          if the metric can be computed just with the values available in the samples array. 
          Default=None.

        - conditions: integer array indicating the condition of each sample (in the same order as
          labels and samples). Default=None.
        
        - num_bootstraps: number of bootstraps sets to create. Default=1000.
        
        - alpha: the confidence interval will be computed between alpha/2 and 100-alpha/2 
          percentiles. Default=5.
        
        - samples2: second set of samples for metrics that require an extra input. Default=None.

        See https://github.com/luferrer/ConfidenceIntervals for more details. 
    """    
    center = metric_wrapper(labels, samples, samples2, metric)

    bt = Bootstrap(num_bootstraps, metric)
    ci = bt.get_conf_int(samples, labels, conditions, alpha=alpha, samples2=samples2)
    
    return center, ci



def get_bootstrap_indices(num_samples, conditions=None, random_state=None):
    """ Method that returns the indices for selecting a bootstrap set.
    - num_samples: number of samples in the original set
    - conditions: integer array indicating the condition of each of those samples (in order)
    - random_state: random state for sampling
    If conditions is None, the indices are obtained by sampling an array from 0 to num_samples-1 with 
    replacement. If conditions is not None, the indices are obtained by sampling conditions first
    and then sampling the indices corresponding to the selected conditions. This code is somewhat 
    slow when the number of conditions is large (the slow part is sampling for each condition).
    """

    indices = np.arange(num_samples)
    if conditions is not None:
        # First sample conditions
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(unique_conditions), 
                                 random_state=random_state)
        
        # Now, for each unique condition selected, sample its indices and repeat them as many times
        # as that condition was selected.
        sel_indices = []
        for s, c in np.c_[np.unique(bt_conditions, return_counts=True)]:
            cond_indices = indices[conditions == s]
            bt_samples_for_cond = resample(cond_indices, replace=True, n_samples=len(cond_indices),
                                           random_state=random_state)   
            sel_indices.append(np.repeat(bt_samples_for_cond, c))
        sel_indices = np.concatenate(sel_indices)
    else:
        sel_indices = resample(indices, replace=True, n_samples=num_samples, random_state=random_state)
        
    return sel_indices


def get_conf_int(values, alpha=5):
        """ Method to obtain the confidence interval from an array of metrics obtained from 
        bootstrapping. Alpha is the level of the test. The confidence interval is computed between 
        alpha/2 and 100-alpha/2 percentiles
        """

        low = np.percentile(values, alpha/2)
        high = np.percentile(values, 100-alpha/2)
        
        return (low, high)

class Bootstrap:

    def __init__(self, num_bootstraps=1000, metric=None):
        """ Class to compute confidence intervals for a metric (e.g. accuracy) using bootstrapping
        - num_bootstraps: number of bootstraps to perform
        - metric: function that takes as input labels and samples and returns a scalar
        """
        self.num_bootstraps = num_bootstraps
        if metric is None:
            self.metric = accuracy_score
        else:
            self.metric = metric

    def get_bootstrap_sets(self, n_samples, conditions=None):
        """ Method to get a list of bootstrap sets. Each set is given by a lists of indices. 
        - n_samples: number of samples in the original set
        - conditions: integer array indicating the condition of each of those samples (in order)
        """
        self.conditions = conditions
        self._indices = []

        for i in range(self.num_bootstraps):
            sel_indices = get_bootstrap_indices(n_samples, self.conditions, random_state=i)
            self._indices.append(sel_indices)

    def get_metric_values_for_bootstrap_sets(self, samples, labels, samples2=None):
        """ Method that computes the metric value for each bootstrap set in self._indices
        - samples: array of decisions/scores/losses for each sample
        - labels: array of labels or any other per-sample information needed to compute the metric 
          This input can be None in which case the metric function is run with samples as the only
          input argument.
        """
        self.samples = samples
        self.samples2 = samples2
        self.labels = labels

        vals = np.zeros(self.num_bootstraps)
        for i, indices in enumerate(self._indices):
            vals[i] = metric_wrapper(self.labels, self.samples, self.samples2, self.metric, indices)
    
        self._scores = vals
        return vals

    def run(self, samples, labels, conditions=None, samples2=None):
        """ Method to compute the confidence interval for the given metric
        - samples: array of decisions for each sample
        - labels: array of labels (0 or 1) for each sample
        - conditions: integer array indicating the condition of each sample (in order)
        """        
        self.get_bootstrap_sets(len(samples), conditions)
        return self.get_metric_values_for_bootstrap_sets(samples, labels, samples2)

    
    def get_conf_int(self, samples, labels, conditions=None, alpha=5, samples2=None):
        """ Method to obtain the confidence interval from an array of metrics obtained from bootstrapping
        """
        vals = self.run(samples, labels, conditions, samples2)
        self._ci = get_conf_int(vals, alpha)
        return self._ci


###################
# wrapper functions
###################
import sandbox.eval_asvspoof as eval_asvspoof

def compute_eer_wrapper(labels, scores, label_true=True):
    """eer = compute_eer_wrapper(labels, scores)
    
    To be usable for the confidence interval API, we need
    to make a compatible wrapper for each metric.
    
    This is the wrapper for EER

    input
    -----
      labels: np.array, (N, ), array of label
      scores: np.array, (N, ), array of scores
      label_true: scalar, label value of positive class, default True

    output
    ------
      eer: scalar

    """
    target_scores = scores[labels == label_true]
    nontar_scores = scores[labels != label_true]
    eer, _ = eval_asvspoof.compute_eer(target_scores, nontar_scores)
    return eer

def get_eer_conf(scores, labels, label_true=True, 
                 conditions=None, num_bootstraps=1000, alpha=5):
    """center, ci = compute_eer_wrapper(scores, labels, 
          label_true=True, conditions=None, num_bootstraps=1000, alpha=5)
    
    This is the wrapper for EER
    it should used as (center - ci, center + ci)

    input
    -----
      scores: np.array, (N, ), array of scores
      labels: np.array, (N, ), array of label
      label_true: scalar, label value of positive class, default True
      conditions: np.array, (N, ), array of integer, indicating the condition
      num_boostraps: int, default 1000
      alpha: int, (0-100), default 5

    output
    ------
      center: scalar mean value of metric
      ci: scalar, conference interval

    """
    # compute the results
    eer, ci = evaluate_with_conf_int(scores, compute_eer_wrapper, labels, 
                                     conditions, num_bootstraps, alpha)
    # 
    return eer, (ci[1] - ci[0])/2.0
    

if __name__ == "__main__":

    # example usage
    scores = np.random.randn(5000)
    scores[0:2500] = scores[0:2500] - 1.5
    scores[2500] = scores[2500] + 1.5

    labels = np.full(5000, True)
    labels[0:2500] = False

    conditions = np.full(5000, True)
    conditions[0:500] = 1
    conditions[500:2500] = 2
    conditions[2500:3000] = 1
    conditions[3000:] = 2

    evaluate_with_conf_int(scores, compute_eer_wrapper, labels, 
                           conditions, num_bootstraps=1000, alpha=5)
    
