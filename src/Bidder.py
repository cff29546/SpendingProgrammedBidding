import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Impression import ImpressionOpportunity
from Models import BidShadingContextualBandit, BidShadingPolicy, PyTorchWinRateEstimator


class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False # Default

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        pass

    def clear_logs(self, memory):
        pass


class TruthfulBidder(Bidder):
    """ A bidder that bids truthfully """
    def __init__(self, rng):
        super(TruthfulBidder, self).__init__(rng)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        return value * estimated_CTR


class BudgetRistrictedBidder(Bidder):
    """ A bidder with budget ristriction """
    def __init__(self, rng, budget, rounds_per_iter):
        super(BudgetRistrictedBidder, self).__init__(rng)
        self.budget = budget
        self.spending = 0

    def charge(self, price, round):
        self.spending += price

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        #print('update {}'.format(contexts is None))
        self.spending = 0


class TruthfulBudgetRistricctedBidder(BudgetRistrictedBidder):
    """ A simple bidder with budget ristriction """
    def __init__(self, rng, budget, rounds_per_iter):
        super(TruthfulBudgetRistricctedBidder, self).__init__(rng, budget, rounds_per_iter)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        if self.spending < self.budget:
            return value * estimated_CTR
        return 0


def aggregate_near_sample(samples, distance=1e-6):
    result = []
    current = []
    for s in samples:
        if len(current) == 0 or s[0] < current[0][0] * (1.0 + distance):
            current.append(s)
        else:
            result.append(list(map(np.mean, zip(*current))))
            current = []

    if len(current) > 0:
        result.append(list(map(np.mean, zip(*current))))

    return result

def increasing_subsequence(samples):
    result = []
    for s in samples:
        l = 0
        r = len(result)
        while l < r:
            mid = (l + r) // 2
            if s[1] >= result[mid][1]:
                l = mid + 1
            else:
                r = mid
        if l == len(result):
            result.append(s)
        else:
            result[l] = s
    return result


def liner_solve(x1, y1, x2, y2, x):
    if np.abs(x1 - x2) < 1e-6:
        return (y1 + y2) / 2
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def impc(bid2spend, target):
    samples = sorted(bid2spend)
    samples = aggregate_near_sample(samples)
    samples = increasing_subsequence(samples)
    if len(samples) == 0:
        return 1.0
    bid = 0
    spend = 0
    i = 0
    while i < len(samples) and samples[i][1] < target:
        bid = samples[i][0]
        spend = samples[i][1]
        i += 1
    if i < len(samples):
        return liner_solve(spend, bid, samples[i][1], samples[i][0], target)
    else:
        return liner_solve(0, 0, samples[-1][1], samples[-1][0], target)


class IMPCBudgetBidder(BudgetRistrictedBidder):
    """ IMPC Budget pacing bidder """
    def __init__(self, rng, budget, rounds_per_iter, rounds_per_step, bid_step, memory):
        super(IMPCBudgetBidder, self).__init__(rng, budget, rounds_per_iter)
        self.round_per_step = rounds_per_step
        self.target_step_spending = budget * rounds_per_step / rounds_per_iter
        self.bid2spend_history = []
        self.bid_step = bid_step
        self.roi_bid = 1.0
        self.step_spending = 0
        self.memory = memory

    def charge(self, price, cur_round):
        self.spending += price
        self.step_spending += price
        if cur_round % self.round_per_step == 0:
            self.bid2spend_history.append([self.roi_bid, self.step_spending])
            self.step_spending = 0
            bid = impc(self.bid2spend_history, self.target_step_spending)
            self.roi_bid = np.minimum(np.maximum(bid, self.roi_bid - self.bid_step), self.roi_bid + self.bid_step)
        self.bid2spend_history = self.bid2spend_history[-self.memory:]

    def bid(self, value, context, estimated_CTR):
        if self.spending < self.budget:
            return value * estimated_CTR * self.roi_bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        print('update {}'.format(contexts is None))
        self.spending = 0


class BidCapBidder(IMPCBudgetBidder):
    """ IMPC Budget pacing bidder """
    def charge(self, price, cur_round):
        super(BidCapBidder, self).charge(price, cur_round)
        if self.roi_bid > 1.0:
            self.roi_bid = 1.0

