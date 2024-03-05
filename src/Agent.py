import numpy as np

from BidderAllocation import PyTorchLogisticRegressionAllocator, OracleAllocator
from Impression import ImpressionOpportunity
from Models import sigmoid
from Bidder import BudgetRistrictedBidder


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, num_items, item_values, allocator, bidder, memory=0, postback_delay=0):
        self.rng = rng
        self.name = name
        self.num_items = num_items

        # Value distribution
        self.item_values = item_values

        self.net_utility = .0
        self.gross_utility = .0
        self.spending = .0

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder

        self.memory = memory
        self.postback_delay = postback_delay
        self.result_cache = []

    def select_item(self, context):
        # Estimate CTR for all items
        estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        # Pick the best item (according to TS)
        best_item = np.argmax(estim_values)

        # If we do Thompson Sampling, don't propagate the noisy bid amount but bid using the MAP estimate
        if type(self.allocator) == PyTorchLogisticRegressionAllocator and self.allocator.thompson_sampling:
            estim_CTRs_MAP = self.allocator.estimate_CTR(context, sample=False)
            return best_item, estim_CTRs_MAP[best_item]

        return best_item, estim_CTRs[best_item]

    def bid(self, context):
        # First, pick what item we want to choose
        best_item, estimated_CTR = self.select_item(context)

        # Sample value for this item
        value = self.item_values[best_item]

        # Get the bid
        bid = self.bidder.bid(value, context, estimated_CTR)

        # Log what we know so far
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               value=value,
                                               bid=bid,
                                               # These will be filled out later
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False))

        return bid, best_item

    def charge(self, price, second_price, outcome, cur_round):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.net_utility += (last_value - price)
        self.gross_utility += last_value
        self.spending += price
        if hasattr(self.bidder, 'charge'):
            self.bidder.charge(price, cur_round, self.logs[-1].estimated_CTR, self.logs[-1].value)

    def set_price(self, price, cur_round):
        self.logs[-1].set_price(price)
        if hasattr(self.bidder, 'charge'):
            self.bidder.charge(0, cur_round, 0, 0)

    def calc_perf_group(self):
        self.acc_value = .0
        self.acc_spending = .0
        self.under_value = .0
        self.under_spending = .0
        self.violation_value = .0
        self.violation_spending = .0

        if self.spending > 0:
            roi_vs_target = self.gross_utility / self.spending
            if roi_vs_target < 0.8:
                self.violation_value += self.gross_utility
                self.violation_spending += self.spending
            elif isinstance(self.bidder, BudgetRistrictedBidder) and self.spending < self.bidder.budget * 0.9 and roi_vs_target > 1.2:
                self.under_value += self.gross_utility
                self.under_spending += self.spending
            else:
                self.acc_value += self.gross_utility
                self.acc_spending += self.spending

    def update(self, iteration, plot=False, figsize=(8,5), fontsize=14):
        # Gather relevant logs
        contexts = np.array(list(opp.context for opp in self.logs))
        items = np.array(list(opp.item for opp in self.logs))
        values = np.array(list(opp.value for opp in self.logs))
        bids = np.array(list(opp.bid for opp in self.logs))
        prices = np.array(list(opp.price for opp in self.logs))
        outcomes = np.array(list(opp.outcome for opp in self.logs))
        sum_values = np.sum(np.array(list(opp.outcome * opp.value for opp in self.logs)))
        estimated_CTRs = np.array(list(opp.estimated_CTR for opp in self.logs))

        # Update response model with data from winning bids
        won_mask = np.array(list(opp.won for opp in self.logs))
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], iteration, plot, figsize, fontsize, self.name)

        #self.result_cache.append([contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name])
        self.result_cache.append([contexts, values, bids, prices, sum_values, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name])
        # Update bidding model with all data
        # self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name)
        if len(self.result_cache) > self.postback_delay:
            self.bidder.update(*self.result_cache[-self.postback_delay-1])
        else:
            self.bidder.update(None, None, None, None, None, None, None, None, plot, figsize, fontsize, self.name)


    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.sum(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs))

    def get_estimation_regret(self):
        ''' How much am I overpaying due to over-estimation of the value? '''
        return np.sum(list(opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs))

    def get_overbid_regret(self):
        ''' How much am I overpaying because I could shade more? '''
        return np.sum(list((opp.price - opp.second_price) * opp.won for opp in self.logs))

    def get_underbid_regret(self):
        ''' How much have I lost because I could have shaded less? '''
        # The difference between the winning price and our bid -- for opportunities we lost, and where we could have won without overpaying
        # Important to mention that this assumes a first-price auction! i.e. the price is the winning bid
        return np.sum(list((opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs))

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean(list((opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs)))

    def get_CTR_bias(self):
        return np.mean(list((opp.estimated_CTR / opp.true_CTR) for opp in filter(lambda opp: opp.won, self.logs)))

    def clear_utility(self):
        self.net_utility = .0
        self.gross_utility = .0
        self.spending = .0

    def clear_logs(self):
        if not self.memory:
            self.logs = []
        else:
            self.logs = self.logs[-self.memory:]
        self.bidder.clear_logs(memory=self.memory)

    def reset_run(self):
        self.result_cache = []
        if hasattr(self.bidder, 'reset'):
            self.bidder.reset()
