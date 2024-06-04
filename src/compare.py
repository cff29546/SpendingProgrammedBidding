import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

FIGSIZE = (8, 5)
FONTSIZE = 14
E1 = '\u03b51'
output_dir = '.'
fmt = 'pdf'
#sns.set_palette([(0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5)])

name_map = {
    'Bid Cap Bidder': 'BidCap',
    'MPC Bidder': 'MPC',
    'SPB Bidder': 'SPB',
}

def plot_measure_over_dim(df, measure_name, dim, cumulative=False, log_y=False, yrange=None, optimal=None):
    fig, axes = plt.subplots(figsize=FIGSIZE)
    #plt.title(f'{measure_name} Over {dim}', fontsize=FONTSIZE + 2, y=-0.2)
    min_measure, max_measure = 0.0, 0.0
    sns.lineplot(data=df, x=dim, y=measure_name, hue='Agent', ax=axes
            #,palette=['0', '0', '0'], style='Agent', style_order=sorted(name_map.values()), dashes=[(1, 1), (4, 2), (1, 0)], markers=True
            )
    plt.xticks(np.arange(df[dim].min(), df[dim].max() + 0.1, 0.1), fontsize=FONTSIZE - 2)
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
    #handles, labels = plt.gca().get_legend_handles_labels()
    #order = list(reversed(range(len(labels))))
    plt.legend(
            #[handles[idx] for idx in order],[labels[idx] for idx in order],
            loc='lower center', bbox_to_anchor=(.5, 1), fontsize=FONTSIZE,
        ncol=3, title=None, frameon=False, handlelength=3, handleheight=1)
    plt.tight_layout()
    delay_info = ''
    dmax = df['Delay'].max()
    dmin = df['Delay'].min()
    if dmax == dmin:
        delay_info = f'_delay_{int(dmax)}'
    else:
        delay_info = f'_delay_{int(dmin)}_{int(dmax)}'
    if dim == E1:
        dim = 'e1'
    dim = dim.replace(' ', '_')
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_{dim}{delay_info}_compare.{fmt}", bbox_inches='tight')
    plt.close()
    return df


def plot_measure_over_dim_hist(df, measure_name, dim, cumulative=False, log_y=False, yrange=None, optimal=None):
    fig, axes = plt.subplots(figsize=FIGSIZE)
    #plt.title(f'{measure_name} Over {dim}', fontsize=FONTSIZE + 2, y=-0.2)
    min_measure, max_measure = 0.0, 0.0
    #sns.lineplot(data=df, x=dim, y=measure_name, hue='Agent', ax=axes)
    sns.histplot(data=df, x=dim, weights=measure_name, bins=list(np.arange(-0.05,2.55,0.1)), kde=False,
            hue='Agent', ax=axes, element='bars', multiple='dodge', shrink=.5) 
    plt.xticks(np.arange(df[dim].min(), df[dim].max() + 0.5, 0.5), fontsize=FONTSIZE - 2)
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
    sns.move_legend(axes, loc='lower center', bbox_to_anchor=(.5, 1), fontsize=FONTSIZE, ncol=3, title=None, frameon=False, handlelength=0.7)
    plt.tight_layout()
    delay_info = ''
    dmax = df['Delay'].max()
    dmin = df['Delay'].min()
    if dmax == dmin:
        delay_info = f'_delay_{int(dmax)}'
    else:
        delay_info = f'_delay_{int(dmin)}_{int(dmax)}'
    if dim == E1:
        dim = 'e1'
    dim = dim.replace(' ', '_')
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_{dim}{delay_info}_compare.{fmt}", bbox_inches='tight')
    plt.close()
    return df


def plot_measure_over_delay(df, measure_name, cumulative=False, log_y=False, yrange=None, optimal=None):
    fig, axes = plt.subplots(figsize=FIGSIZE)
    #plt.title(f'{measure_name} Over Delay Level', fontsize=FONTSIZE + 2, y=-0.2)
    min_measure, max_measure = 0.0, 0.0
    sns.histplot(data=df, x='Delay', weights=measure_name, discrete=True, hue='Agent', element='bars',
            multiple='dodge', shrink=.5, ax=axes)
    plt.xticks(range(df['Delay'].max() + 1),fontsize=FONTSIZE - 2)
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
    sns.move_legend(axes, loc='lower center', bbox_to_anchor=(.5, 1), fontsize=FONTSIZE, ncol=3, title=None, frameon=False, handlelength=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{measure_name.replace(' ', '_')}_over_delay_level_compare.{fmt}", bbox_inches='tight')
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
    parser.add_argument('-f', '--format', type=str, default='pdf', choices=['pdf', 'png'])
    parser.add_argument('bidders', nargs='*', default=[])
    args = parser.parse_args()

    output_dir = os.path.join(args.output, 'compare_' + '.'.join(map(os.path.basename, args.bidders)))
    # Make sure we can write results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fmt = args.format

    e1df = concat_csv(args.bidders, 'e1.csv')
    e1df['Agent'] = e1df['Agent'].apply(lambda name: name_map.get(name, name))
    for i in range(e1df['Delay'].max() + 1):
        plot_measure_over_dim(e1df[e1df['Delay'] == i], 'Violation Value Ratio', E1)
    roidf = concat_csv(args.bidders, 'roi.csv')
    roidf['Agent'] = roidf['Agent'].apply(lambda name: name_map.get(name, name))
    for i in range(roidf['Delay'].max() + 1):
        plot_measure_over_dim_hist(roidf[roidf['Delay'] == i], 'Value Ratio', 'Roi vs Target Roi')
    df = concat_csv(args.bidders, 'value_spend.csv')
    df['Agent'] = df['Agent'].apply(lambda name: name_map.get(name, name))

    keys = [
        'Total Value',
        'Total Spending',
        'Accomplish Value',
        'Accomplish Spending',
        'Violation Value',
        'Violation Spending',
        'Under Performance Value',
        'Under Performance Spending'
    ]
    sumdf = df.groupby(['Agent', 'Delay'])[keys].sum().reset_index()
    for key in keys:
        sumdf[f'{key} Ratio'] = sumdf[key]/sumdf['Total Value']
    print(sumdf)

    for key in keys:
        plot_measure_over_delay(sumdf, key)
        if not key.startswith('Total'):
            plot_measure_over_delay(sumdf, f'{key} Ratio')

