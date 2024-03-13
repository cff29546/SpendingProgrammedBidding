import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

FIGSIZE = (8, 5)
FONTSIZE = 14
output_dir = '.'

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
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_{dim}_compare.pdf", bbox_inches='tight')
    plt.close()
    return df

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
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_runs_compare.pdf", bbox_inches='tight')
    plt.close()
    return df

def concat_csv(bidders, csv):
    df = None
    for bidder in bidders:
        if df is not None:
            df = pd.concat((df, pd.read_csv(os.path.join(bidder, csv))), ignore_index=True)
        else:
            df = pd.read_csv(os.path.join(bidder, csv))
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='bidder compare')
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('bidders', nargs='*', default=[])
    args = parser.parse_args()

    output_dir = os.path.join(args.output, 'compare_' + '.'.join(map(os.path.basename, args.bidders)))
    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    e1df = concat_csv(args.bidders, 'e1.csv')
    plot_measure_over_dim(e1df, 'Violation Value Rate', 'e1')
    roidf = concat_csv(args.bidders, 'roi.csv')
    plot_measure_over_dim(roidf, 'Value Rate', 'Roi vs Target Roi')
    df = concat_csv(args.bidders, 'value_spend.csv')

    plot_measure_over_runs(df, "Accomplish Value Rate")
    #plot_measure_over_runs(df, "Under Performance Value Rate")
    #plot_measure_over_runs(df, "Violation Value Rate")
    #plot_measure_over_runs(df, "Accomplish Spending Rate")
    #plot_measure_over_runs(df, "Under Performance Spending Rate")
    #plot_measure_over_runs(df, "Violation Spending Rate")

    plot_measure_over_runs(df, "Accomplish Value")
    #plot_measure_over_runs(df, "Under Performance Value")
    #plot_measure_over_runs(df, "Violation Value")
    #plot_measure_over_runs(df, "Accomplish Spending")
    #plot_measure_over_runs(df, "Under Performance Spending")
    #plot_measure_over_runs(df, "Violation Spending")


