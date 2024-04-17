#!/usr/bin/env python
"""
This module is based on Cllr and min Cllr implemented by Dr. Andreas Nautsch

https://gitlab.eurecom.fr/nautsch/cllr

license: LGPLv3 
version: 2020-01-10 
author: Andreas Nautsch (EURECOM)
"""

import numpy
import copy

def logit(p):
    """logit function.
    This is a one-to-one mapping from probability to log-odds.
    i.e. it maps the interval (0,1) to the real line.
    The inverse function is given by SIGMOID.

    log_odds = logit(p) = log(p/(1-p))

    :param p: the inumpyut value

    :return: logit(inumpyut)
    """
    p = numpy.array(p)
    lp = numpy.zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = numpy.log(p / (1 - p))
        elif f0:
            lp = -numpy.inf
        elif f1:
            lp = numpy.inf
    else:
        lp[f] = numpy.log(p[f] / (1 - p[f]))
        lp[f0] = -numpy.inf
        lp[f1] = numpy.inf
    return lp


def sigmoid(log_odds):
    """SIGMOID: Inverse of the logit function.
    This is a one-to-one mapping from log odds to probability.
    i.e. it maps the real line to the interval (0,1).

    p = sigmoid(log_odds)

    :param log_odds: the inumpyut value

    :return: sigmoid(inumpyut)
    """
    p = 1 / (1 + numpy.exp(-log_odds))
    return p

def pavx(y):
    """PAV: Pool Adjacent Violators algorithm.
    Non-paramtetric optimization subject to monotonicity.

    ghat = pav(y)
    fits a vector ghat with nondecreasing components to the
    data vector y such that sum((y - ghat).^2) is minimal.
    (Pool-adjacent-violators algorithm).

    optional outputs:
            width: width of pav bins, from left to right
                    (the number of bins is data dependent)
            height: corresponding heights of bins (in increasing order)

    Author: This code is a simplified version of the 'IsoMeans.m' code
    made available by Lutz Duembgen at:
    http://www.imsv.unibe.ch/~duembgen/software

    :param y: inumpyut value
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Inumpyut array is empty'
    n = y.shape[0]

    index = numpy.zeros(n, dtype=int)
    length = numpy.zeros(n, dtype=int)

    ghat = numpy.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[numpy.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[:ci + 1])
    width = copy.deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n + 1):
            ghat[j - 1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def optimal_llr(tar, non, laplace=False, 
                monotonicity_epsilon=1e-6, compute_eer=False):
    """
    
    input
    -----
      tar_llrs:    np.array (N, ), null hypothesis LLRs
      nontar_llrs: np.array (M, ), alter hypothesis LLRs
      laplace:     bool, 
    
    output
    ------
      c: scalar, Cllr value

    Comment by original author: 
    Laplace flag avoids infinite LLR magnitudes;
    also, this stops DET cureves from 'curling' to the axes on sparse data 
    (DETs stay in more populated regions)
    """

    scores = numpy.concatenate([non, tar])
    Pideal = numpy.concatenate([numpy.zeros(len(non)), numpy.ones(len(tar))])

    perturb = numpy.argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
        Pideal = numpy.hstack([1, 0, Pideal, 1, 0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
        Popt = Popt[2:len(Popt) - 2]

    posterior_log_odds = logit(Popt)
    log_prior_odds = numpy.log(len(tar) / len(non))
    llrs = posterior_log_odds - log_prior_odds
    N = len(tar) + len(non)
    llrs = llrs + numpy.arange(N) * monotonicity_epsilon / N  # preserve monotonicity

    idx_reverse = numpy.zeros(len(scores), dtype=int)
    idx_reverse[perturb] = numpy.arange(len(scores))
    tar_llrs = llrs[idx_reverse][len(non):]
    nontar_llrs = llrs[idx_reverse][:len(non)]

    if not compute_eer:
        return tar_llrs, nontar_llrs

    nbins = width.shape[0]
    pmiss = numpy.zeros(nbins + 1)
    pfa = numpy.zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = non.shape[0]
    miss = 0
    #
    for i in range(nbins):
        pmiss[i] = miss / len(tar)
        pfa[i] = fa /len(non)
        left = int(left + width[i])
        miss = numpy.sum(Pideal[:left])
        fa = len(tar) + len(non) - left - numpy.sum(Pideal[left:])
    #
    pmiss[nbins] = miss / len(tar)
    pfa[nbins] = fa / len(non)

    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = numpy.column_stack((xx, yy))
        dd = numpy.dot(numpy.array([1, -1]), XY)
        if numpy.min(numpy.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = numpy.linalg.solve(XY, numpy.array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (numpy.sum(seg))

        eer = max([eer, eerseg])
    return tar_llrs, nontar_llrs, eer




def compute_cllr(tar_llrs, nontar_llrs):
    """cllr = compute_cllr(tar_llrs, nontar_llrs)
    
    input
    -----
      tar_llrs:    np.array (N, ), null hypothesis LLRs
      nontar_llrs: np.array (M, ), alter hypothesis LLRs

    output
    ------
      c: scalar, Cllr value
    """
    log2 = numpy.log(2)

    tar_posterior = sigmoid(tar_llrs)
    non_posterior = sigmoid(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return numpy.inf

    c1 = (-numpy.log(tar_posterior)).mean() / log2
    c2 = (-numpy.log(non_posterior)).mean() / log2
    c = (c1 + c2) / 2
    return c


def compute_min_cllr(tar_llrs, nontar_llrs, 
                     monotonicity_epsilon=1e-6, 
                     compute_eer=False):
    """cllr = compute_min_cllr(tar_llrs, nontar_llrs)
    
    Compute min Cllr, where calibration is done by PAV

    input
    -----
      tar_llrs:    np.array (N, ), null hypothesis LLRs
      nontar_llrs: np.array (M, ), alter hypothesis LLRs

    output
    ------
      c: scalar, Cllr value
    """
    if tar_llrs.size < 1 or nontar_llrs.size < 1:
        if compute_eer:
            return numpy.nan, numpy.nan
        else:
            return numpy.nan

    if compute_eer:
        [tar, non, eer] = optimal_llr(tar_llrs, 
                                      nontar_llrs, 
                                      laplace=False, 
                                      monotonicity_epsilon=monotonicity_epsilon,
                                      compute_eer=compute_eer)
        cmin = compute_cllr(tar, non)
        return cmin, eer
    else:
        [tar,non] = optimal_llr(tar_llrs, 
                                nontar_llrs, 
                                laplace=False, 
                                monotonicity_epsilon=monotonicity_epsilon)
        cmin = compute_cllr(tar, non)
        return cmin



if __name__ == "__main__":
    print(__doc__)
