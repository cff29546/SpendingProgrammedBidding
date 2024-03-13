import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from Agent import Agent
from AuctionAllocation import * # FirstPrice, SecondPrice
from Auction import Auction
from Bidder import *  # EmpiricalShadedBidder, TruthfulBidder
from BidderAllocation import *  #  LogisticTSAllocator, OracleAllocator


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Number of runs
    num_runs = config['num_runs'] if 'num_runs' in config.keys() else 1

    # Max. number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    agent_configs, agents2items, agents2item_values = rerandom(rng, config)

    return rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size


def rerandom(rng, config):
    # Technical parameters for distribution of latent embeddings
    embedding_size = config['embedding_size']
    embedding_var = config['embedding_var']
    obs_embedding_size = config['obs_embedding_size']

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config['agents']:
        if 'num_copies' in agent_config.keys():
            for i in range(1, agent_config['num_copies'] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy['name'] += f' {num_agents + 1}'
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config['name']: rng.normal(0.0, embedding_var, size=(agent_config['num_items'], embedding_size))
        for agent_config in agent_configs
    }

    agents2item_values = {
        agent_config['name']: rng.lognormal(0.1, 0.2, agent_config['num_items'])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    for agent, items in agents2items.items():
        agents2items[agent] = np.hstack((items, - 3.0 - 1.0 * rng.random((items.shape[0], 1))))

    return agent_configs, agents2items, agents2item_values


def instantiate_agents(rng, agent_configs, agents2item_values, agents2items):
    # Store agents to be re-instantiated in subsequent runs
    # Set up agents
    agents = [
        Agent(rng=rng,
              name=agent_config['name'],
              num_items=agent_config['num_items'],
              item_values=agents2item_values[agent_config['name']],
              allocator=eval(f"{agent_config['allocator']['type']}(rng=rng{parse_kwargs(agent_config['allocator']['kwargs'])})"),
              bidder=eval(f"{agent_config['bidder']['type']}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})"),
              memory=(0 if 'memory' not in agent_config.keys() else agent_config['memory']),
              postback_delay=int(agent_config.get('postback_delay', 0)))
        for agent_config in agent_configs
    ]

    for agent in agents:
        if isinstance(agent.allocator, OracleAllocator) or isinstance(agent.allocator, IsotonicPerturbationOracleAllocator):
            agent.allocator.update_item_embeddings(agents2items[agent.name])

    return agents


def instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size):
    return (Auction(rng,
                    eval(f"{config['allocation']}()"),
                    agents,
                    agents2items,
                    agents2item_values,
                    max_slots,
                    embedding_size,
                    embedding_var,
                    obs_embedding_size,
                    config['num_participants_per_round']),
            config['num_iter'], config['rounds_per_iter'], config['output_dir'])


def simulation_run():

    for i in range(num_iter):
        print(f'==== ITERATION {i} ====')

        for _ in tqdm(range(rounds_per_iter)):
            auction.simulate_opportunity()

        names = [agent.name for agent in auction.agents]
        net_utilities = [agent.net_utility for agent in auction.agents]
        gross_utilities = [agent.gross_utility for agent in auction.agents]

        for agent in auction.agents:
            agent.calc_perf_group()
        acc_value = [agent.acc_value for agent in auction.agents]
        acc_spending = [agent.acc_spending for agent in auction.agents]
        under_value = [agent.under_value for agent in auction.agents]
        under_spending = [agent.under_spending for agent in auction.agents]
        violation_value = [agent.violation_value for agent in auction.agents]
        violation_spending = [agent.violation_spending for agent in auction.agents]

        result = pd.DataFrame({
            'Name': names,
            #'Net': net_utilities,
            #'Gross': gross_utilities,
            'Acc': acc_value,
            'Under': under_value,
            'Violation': violation_value,
            'Acc Spend': acc_spending,
            'Under Spend': under_spending,
            'Violation Spend': violation_spending,
            })

        print(result)
        #print(f'\tAuction revenue: \t {auction.revenue}')

        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i, plot=True, figsize=FIGSIZE, fontsize=FONTSIZE)

            agent2net_utility[agent.name].append(agent.net_utility)
            agent2gross_utility[agent.name].append(agent.gross_utility)
            
            agent2spending[agent.name].append(agent.spending)
            agent2budget[agent.name].append(agent.budget)

            agent2allocation_regret[agent.name].append(agent.get_allocation_regret())
            agent2estimation_regret[agent.name].append(agent.get_estimation_regret())
            agent2overbid_regret[agent.name].append(agent.get_overbid_regret())
            agent2underbid_regret[agent.name].append(agent.get_underbid_regret())

            #agent2CTR_RMSE[agent.name].append(agent.get_CTR_RMSE())
            #agent2CTR_bias[agent.name].append(agent.get_CTR_bias())

            #if not agent.bidder.truthful:
            #    agent2gamma[agent.name].append(np.mean(agent.bidder.gammas))

            #best_expected_value = np.mean([opp.best_expected_value for opp in agent.logs])
            #agent2best_expected_value[agent.name].append(best_expected_value)

            #print('Average Best Value for Agent: ', best_expected_value)
            agent.clear_utility()
            agent.clear_logs()

        auction_revenue.append(auction.revenue)
        auction.clear_revenue()
        
    for agent_id, agent in enumerate(auction.agents):
        agent.reset_run()

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    args = parser.parse_args()

    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, obs_embedding_size = parse_config(args.config)

    # Plotting config
    FIGSIZE = (8, 5)
    FONTSIZE = 14

    # Placeholders for summary statistics over all runs
    run2agent2net_utility = {}
    run2agent2gross_utility = {}
    run2agent2allocation_regret = {}
    run2agent2estimation_regret = {}
    run2agent2overbid_regret = {}
    run2agent2underbid_regret = {}
    run2agent2best_expected_value = {}
    
    run2agent2spending = {}
    run2agent2budget = {}

    #run2agent2CTR_RMSE = {}
    #run2agent2CTR_bias = {}
    run2agent2gamma = {}

    run2auction_revenue = {}

    # Repeated runs
    for run in range(num_runs):
        # Rerandomize campaign
        agent_configs, agents2items, agents2item_values = rerandom(rng, config)
        # Reinstantiate agents and auction per run
        agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)
        auction, num_iter, rounds_per_iter, output_dir = instantiate_auction(rng, config, agents2items, agents2item_values, agents, max_slots, embedding_size, embedding_var, obs_embedding_size)

        # Placeholders for summary statistics per run
        agent2net_utility = defaultdict(list)
        agent2gross_utility = defaultdict(list)
        agent2allocation_regret = defaultdict(list)
        agent2estimation_regret = defaultdict(list)
        agent2overbid_regret = defaultdict(list)
        agent2underbid_regret = defaultdict(list)
        agent2best_expected_value = defaultdict(list)

        agent2spending = defaultdict(list)
        agent2budget = defaultdict(list)

        agent2CTR_RMSE = defaultdict(list)
        agent2CTR_bias = defaultdict(list)
        agent2gamma = defaultdict(list)

        auction_revenue = []

        # Run simulation (with global parameters -- fine for the purposes of this script)
        simulation_run()

        # Store
        run2agent2net_utility[run] = agent2net_utility
        run2agent2gross_utility[run] = agent2gross_utility
        run2agent2allocation_regret[run] = agent2allocation_regret
        run2agent2estimation_regret[run] = agent2estimation_regret
        run2agent2overbid_regret[run] = agent2overbid_regret
        run2agent2underbid_regret[run] = agent2underbid_regret
        run2agent2best_expected_value[run] = agent2best_expected_value

        run2agent2spending[run] = agent2spending
        run2agent2budget[run] = agent2budget

        #run2agent2CTR_RMSE[run] = agent2CTR_RMSE
        #run2agent2CTR_bias[run] = agent2CTR_bias
        run2agent2gamma[run] = agent2gamma

        run2auction_revenue[run] = auction_revenue

    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def perf_group(value, spending, budget, e1, e2, e3):
        target = 1.0
        if spending > 0 and budget > 0:
            roi_vs_target = value / spending / target
            if roi_vs_target < (1.0 - e1):
                return 'Violation'
            elif spending < budget * (1.0 - e3) and roi_vs_target > (1.0 + e2):
                return 'Under Performance'
            else:
                return 'Accomplish'
        return None

    def roi_level(value, spending, cap):
        target = 1.0
        if spending > 0:
            roi_vs_target = value / spending / target
            return f'{roi_vs_target if roi_vs_target < cap else cap:.1f}'
        return f'{cap:.1f}'
            

    df_rows = {'Run': [], 'Agent': []}
 
    e1_range = [f'{e1/10.0:.1f}' for e1 in range(1,6,1)]
    roi_range = [f'{r/10.0:.1f}' for r in range(0,26,1)]
    agent2e1value = {}
    agent2roi2value = {}
    for run in range(num_runs):
        for name in run2agent2gross_utility[run]:
            if name.startswith("Environment"):
                continue
            group_value = {'Total': .0, 'Accomplish': .0, 'Violation': .0, 'Under Performance': .0}
            agent2e1value[name] = agent2e1value.get(name, {})
            agent2roi2value[name] = agent2roi2value.get(name, {})
            group_spending = {'Total': .0, 'Accomplish': .0, 'Violation': .0, 'Under Performance': .0}
            for value, spending, budget in zip(run2agent2gross_utility[run][name], run2agent2spending[run][name], run2agent2budget[run][name]):
                group = perf_group(value, spending, budget, 0.2, 0.2, 0.1)
                if group:
                    group_value[group] += value
                    group_value['Total'] += value
                    group_spending[group] += spending
                    group_spending['Total'] += spending

                agent2e1value[name]['Total'] = agent2e1value[name].get('Total', 0.0) + value
                agent2roi2value[name]['Total'] = agent2roi2value[name].get('Total', 0.0) + value
                for e1 in e1_range:
                    group = perf_group(value, spending, budget, float(e1), 0.2, 0.1)
                    if group == 'Violation':
                        agent2e1value[name][e1] = agent2e1value[name].get(e1, 0.0) + value
                roi = roi_level(value, spending, 2.5)
                agent2roi2value[name][roi] = agent2roi2value[name].get(roi, 0.0) + value

            df_rows['Run'].append(run)
            df_rows['Agent'].append(name)
            for group, value in group_value.items():
                df_rows[group + ' Value'] = df_rows.get(group + ' Value', [])
                df_rows[group + ' Value'].append(value)
                df_rows[group + ' Value Rate'] = df_rows.get(group + ' Value Rate', [])
                df_rows[group + ' Value Rate'].append(value / max(group_value['Total'], 1))
                df_rows[group + ' Spending'] = df_rows.get(group + ' Spending', [])
                df_rows[group + ' Spending'].append(group_spending[group])
                df_rows[group + ' Spending Rate'] = df_rows.get(group + ' Spending Rate', [])
                df_rows[group + ' Spending Rate'].append(group_spending[group] / max(group_spending['Total'], 1))

    df = pd.DataFrame(df_rows)
    e1_rows = {'Agent': [], 'e1': [], 'Violation Value Rate': []}
    for name in agent2e1value:
        total = agent2e1value[name].get('Total', 0.0)
        for e1 in e1_range:
            value = agent2e1value[name].get(e1, 0.0)
            e1_rows['Agent'].append(name)
            e1_rows['e1'].append(float(e1))
            e1_rows['Violation Value Rate'].append(value/total if total > 0 else 0)
    e1df = pd.DataFrame(e1_rows)
    roi_rows = {'Agent': [], 'Roi vs Target Roi': [], 'Value Rate': []}
    for name in agent2roi2value:
        total = agent2roi2value[name].get('Total', 0.0)
        for roi in roi_range:
            value = agent2roi2value[name].get(roi, 0.0)
            roi_rows['Agent'].append(name)
            roi_rows['Roi vs Target Roi'].append(float(roi))
            roi_rows['Value Rate'].append(value/total if total > 0 else 0)
    roidf = pd.DataFrame(roi_rows)

    def plot_measure_over_dim(df, measure_name, dim, cumulative=False, log_y=False, yrange=None, optimal=None):
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over {dim}', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x=dim, y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_{dim}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.pdf", bbox_inches='tight')
        plt.close()
        # plt.show()
        return df

    plot_measure_over_dim(e1df, 'Violation Value Rate', 'e1')
    plot_measure_over_dim(roidf, 'Value Rate', 'Roi vs Target Roi')

    def plot_measure_over_runs(df, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
        fig, axes = plt.subplots(figsize=FIGSIZE)
        plt.title(f'{measure_name} Over Runs', fontsize=FONTSIZE + 2)
        min_measure, max_measure = 0.0, 0.0
        sns.lineplot(data=df, x="Run", y=measure_name, hue="Agent", ax=axes)
        plt.xticks(fontsize=FONTSIZE - 2)
        plt.ylabel(f'{measure_name}', fontsize=FONTSIZE)
        if optimal is not None:
            plt.axhline(optimal, ls='--', color='gray', label='Optimal')
            min_measure = min(min_measure, optimal)
        if log_y:
            plt.yscale('log')
        if yrange is None:
            factor = 1.1 if min_measure < 0 else 0.9
            # plt.ylim(min_measure * factor, max_measure * 1.1)
        else:
            plt.ylim(yrange[0], yrange[1])
        plt.yticks(fontsize=FONTSIZE - 2)
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.legend(loc='upper left', bbox_to_anchor=(-.05, -.15), fontsize=FONTSIZE, ncol=3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_{rounds_per_iter}_rounds_{num_iter}_iters_{num_runs}_runs.pdf", bbox_inches='tight')
        plt.close()
        # plt.show()
        return df

    plot_measure_over_runs(df, "Accomplish Value Rate")
    plot_measure_over_runs(df, "Under Performance Value Rate")
    plot_measure_over_runs(df, "Violation Value Rate")
    plot_measure_over_runs(df, "Accomplish Spending Rate")
    plot_measure_over_runs(df, "Under Performance Spending Rate")
    plot_measure_over_runs(df, "Violation Spending Rate")

    #plot_measure_over_runs(df, "Accomplish Value")
    #plot_measure_over_runs(df, "Under Performance Value")
    #plot_measure_over_runs(df, "Violation Value")
    #plot_measure_over_runs(df, "Accomplish Spending")
    #plot_measure_over_runs(df, "Under Performance Spending")
    #plot_measure_over_runs(df, "Violation Spending")
    


