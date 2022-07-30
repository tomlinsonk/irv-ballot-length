import glob
import os
from collections import defaultdict

import numpy as np
import pickle
import matplotlib.pyplot as plt
import yaml
from matplotlib import patches, cm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config['resultsdir']


def plot_truncation_heatmaps():

    names = ['1d', '1d-partial', 'general', 'general-partial']
    nice_names = ['1-Euclidean', '1-Euclidean partial', 'General', 'General partial']

    for name, nice_name in zip(names, nice_names):

        cp = '-cand-pos' if '1d' in name else ''

        with open(f'results/{name}-truncation-results-40-max-10000-trials{cp}.pickle', 'rb') as f:
            results = pickle.load(f)

        agree_frac = results[-3]

        agree_frac = np.hstack((np.zeros((agree_frac.shape[0], 1)), agree_frac))
        agree_frac = np.vstack((np.zeros((1, agree_frac.shape[1])), agree_frac))

        fig, ax = plt.subplots(figsize=(4, 3))

        cmap = cm.get_cmap('inferno').copy()
        cmap.set_bad('w')

        agree_frac = np.ma.masked_array(agree_frac, mask=agree_frac==0)

        plt.imshow(agree_frac, cmap='inferno', origin='lower')
        plt.xlim(0.5, 40.5)
        plt.ylim(2.5, 40.5)
        plt.colorbar(label='Pr(IRV winner wins)')

        w1 = patches.Wedge((0, 0), 100, 0, 45, fc='white', hatch=r'++')
        w2 = patches.Wedge((0, 0), 100, 0, 45, fc='white', alpha=0.95)

        ax.add_patch(w1)
        ax.add_patch(w2)


        plt.ylabel('# candidates ($k$)')
        plt.xlabel('ballot length ($h$)')
        plt.title(nice_name)
        plt.savefig(f'plots/{name}-ballot-length-heatmap.pdf', bbox_inches='tight')
        plt.close()


def plot_combined_small_heatmaps():
    names = ['general', '1d']
    nice_names = ['General', '1-Euclidean']

    fig, axes = plt.subplots(1, 2, sharey='row', figsize=(6, 2.5))

    for col, (name, nice_name) in enumerate(zip(names, nice_names)):
        cp = '-cand-pos' if '1d' in name else ''

        with open(f'results/{name}-truncation-results-40-max-10000-trials{cp}.pickle', 'rb') as f:
            results = pickle.load(f)

        agree_frac = results[-3]

        agree_frac = np.hstack((np.zeros((agree_frac.shape[0], 1)), agree_frac))
        agree_frac = np.vstack((np.zeros((1, agree_frac.shape[1])), agree_frac))

        cmap = cm.get_cmap('inferno').copy()
        cmap.set_bad('w')

        agree_frac = np.ma.masked_array(agree_frac, mask=agree_frac == 0)

        im = axes[col].imshow(agree_frac, cmap='inferno', origin='lower')
        axes[col].set_xlim(0.5, 40.5)
        axes[col].set_ylim(2.5, 40.5)

        w1 = patches.Wedge((0, 0), 100, 0, 45, fc='white', hatch=r'++')
        w2 = patches.Wedge((0, 0), 100, 0, 45, fc='white', alpha=0.95)

        axes[col].add_patch(w1)
        axes[col].add_patch(w2)

        axes[col].set_xlabel('ballot length ($h$)')
        axes[col].set_title(nice_name)

    axes[0].set_ylabel('# candidates ($k$)')

    fig.subplots_adjust(wspace=0.05, right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.11, 0.02, 0.775])
    fig.colorbar(im, cax=cbar_ax, label='Pr(IRV winner wins)')

    plt.savefig(f'plots/combined-ballot-length-heatmap.pdf', bbox_inches='tight')
    plt.close()


def plot_small_stacked_bars():

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    candidate_counts = []
    ballot_lengths = []
    voter_counts = []
    min_unique_winners = []
    max_unique_winners = []
    true_unique_winners = []
    expected_unique_winners = []

    out_dir = 'plots/preflib-data'
    os.makedirs(out_dir, exist_ok=True)

    resample_win_prob_dir = 'plots/preflib-data/resampling-win-probabilities'
    os.makedirs(resample_win_prob_dir, exist_ok=True)

    replace_names = {'ED-00007-00000005': 'ERS Election 5', 'ED-00005-00000002': '2009 Burlington Mayor'}

    burlington_results = None
    ers_results = None

    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')
        if stripped_election_name == 'ED-00007-00000005':
            ers_results = collection, election_name, ballots, ballot_counts, cand_names, skipped_votes

        elif stripped_election_name == 'ED-00005-00000002':
            burlington_results = collection, election_name, ballots, ballot_counts, cand_names, skipped_votes

    fig, axes = plt.subplots(1, 2, figsize=(6, 2), sharey='row')

    for col, results in enumerate((burlington_results, ers_results)):
        collection, election_name, ballots, ballot_counts, cand_names, skipped_votes = results
        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')

        winners_seqs = np.array([winners for winners, majority_winner in resampled_results[collection, election_name]])

        true_winner_seq = np.array(true_results[collection, election_name][0])

        candidate_counts.append(len(cand_names))
        # print(len(cand_names))
        ballot_lengths.append(max(len(b) for b in ballots))
        voter_counts.append(sum(ballot_counts))

        unique_winners = np.unique(winners_seqs)
        unique_winners = np.array(sorted(unique_winners, key=lambda x: cand_names[x]))

        true_unique_winners.append(len(np.unique(true_winner_seq)))
        unique_winner_counts = [len(np.unique(row)) for row in winners_seqs]
        min_unique_winners.append(min(unique_winner_counts))
        max_unique_winners.append(max(unique_winner_counts))
        expected_unique_winners.append(np.mean(unique_winner_counts))

        proportions = np.zeros((len(unique_winners), winners_seqs.shape[1]))
        for i, winner in enumerate(unique_winners):
            proportions[i] = np.sum(winners_seqs == winner, axis=0) / winners_seqs.shape[0]

        for winner in reversed(range(unique_winners.shape[0])):
            bottom = np.sum(proportions[:winner], axis=0)
            axes[col].bar(np.arange(1, winners_seqs.shape[1] + 1), proportions[winner], bottom=bottom,
                    label=cand_names[unique_winners[winner]], width=1)

            for h, true_winner in enumerate(true_winner_seq):
                if true_winner == unique_winners[winner]:
                    axes[col].scatter([1 + h], [bottom[h] + proportions[winner][h] / 2], marker='*', color='black')

        if collection != 'ers':
            axes[col].legend(fontsize=8, framealpha=0.8)

        axes[col].set_title(f'{replace_names[stripped_election_name]}')

        # plt.xticks(range(1, winners_seqs.shape[1]+1), fontsize=8)
        axes[col].set_xlim(0.5, winners_seqs.shape[1] + 0.5)
        axes[col].set_ylim(0, 1)

        axes[col].set_xlabel('Ballot length $h$')

        if col == 1:
            axes[col].set_xticks([1, 5, 10, 15, 20, 25])

    axes[0].set_ylabel('Resampling win prob.')
    plt.subplots_adjust(wspace=0.1)

    plt.savefig(f'plots/combined-stacked-bars.pdf', bbox_inches='tight')
    plt.close()


def plot_preflib_resampling():
    # Skip duplicate elections with tons of write-ins
    to_skip = ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    candidate_counts = []
    ballot_lengths = []
    voter_counts = []
    min_unique_winners = []
    max_unique_winners = []
    true_unique_winners = []
    expected_unique_winners = []

    out_dir = 'plots/preflib-data'
    os.makedirs(out_dir, exist_ok=True)

    resample_win_prob_dir = 'plots/preflib-data/resampling-win-probabilities'
    os.makedirs(resample_win_prob_dir, exist_ok=True)

    replace_names = {'ED-00007-00000005': 'ERS Election 5', 'ED-00005-00000002': '2009 Burlington Mayor'}

    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        if election_name in to_skip:
            continue

        stripped_election_name = election_name.replace('.soi', '').replace('.toi', '')

        print(collection, election_name)

        winners_seqs = np.array([winners for winners, majority_winner in resampled_results[collection, election_name]])

        true_winner_seq = np.array(true_results[collection, election_name][0])

        candidate_counts.append(len(cand_names))
        ballot_lengths.append(max(len(b) for b in ballots))
        voter_counts.append(sum(ballot_counts))

        unique_winners = np.unique(winners_seqs)
        unique_winners = np.array(sorted(unique_winners, key=lambda x: cand_names[x]))

        true_unique_winners.append(len(np.unique(true_winner_seq)))
        unique_winner_counts = [len(np.unique(row)) for row in winners_seqs]
        min_unique_winners.append(min(unique_winner_counts))
        max_unique_winners.append(max(unique_winner_counts))
        expected_unique_winners.append(np.mean(unique_winner_counts))

        if len(unique_winners) > 1:
            proportions = np.zeros((len(unique_winners), winners_seqs.shape[1]))
            for i, winner in enumerate(unique_winners):
                proportions[i] = np.sum(winners_seqs == winner, axis=0) / winners_seqs.shape[0]

            plt.figure(figsize=(3, 2))
            for winner in reversed(range(unique_winners.shape[0])):
                bottom = np.sum(proportions[:winner], axis=0)
                plt.bar(np.arange(1, winners_seqs.shape[1]+1), proportions[winner], bottom=bottom, label=cand_names[unique_winners[winner]], width=1)

                for h, true_winner in enumerate(true_winner_seq):
                    if true_winner == unique_winners[winner]:
                        plt.scatter([1+h], [bottom[h] + proportions[winner][h] / 2], marker='*', color='black')

            if collection != 'ers':
                plt.legend(fontsize=8, framealpha=0.8)

            if stripped_election_name in replace_names:
                plt.title(f'{replace_names[stripped_election_name]}')
            else:
                plt.title(f'{collection}, {stripped_election_name}, $k={len(cand_names)}$, $n={sum(ballot_counts)}$')
            plt.xlim(0.5, winners_seqs.shape[1]+0.5)
            plt.ylim(0, 1)
            plt.xlabel('Ballot length $h$')
            plt.ylabel('Resampling win prob.')
            plt.savefig(f'{resample_win_prob_dir}/{stripped_election_name}-stacked-bars.pdf', bbox_inches='tight')
            plt.close()


    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)

    labels, counts = np.unique(true_unique_winners, return_counts=True)
    axes[0].bar(labels, counts, align='center')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Truncation winners')
    axes[0].set_title('Actual')

    axes[1].hist(expected_unique_winners, bins=30)
    axes[1].set_xlabel('Truncation winners')
    axes[1].set_title('Expected')

    labels, counts = np.unique(max_unique_winners, return_counts=True)
    axes[2].bar(labels, counts, align='center')
    axes[2].set_xticks([1, 2, 3, 4, 5])
    axes[2].set_xlabel('Truncation winners')
    axes[2].set_title('Maximum')

    # plt.xscale('log')
    plt.savefig(f'{out_dir}/truncation-winners.pdf', bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))

    axes[0].bar(np.arange(max(candidate_counts)+1), np.bincount(candidate_counts), width=1)
    axes[0].set_xlabel('number of candidates $k$')
    axes[0].set_ylabel('count')
    axes[0].set_xlim(left=0)

    axes[1].bar(np.arange(max(ballot_lengths)+1), np.bincount(ballot_lengths), width=1)
    axes[1].set_xlabel('ballot length $h$')
    axes[1].sharey(axes[0])
    axes[1].set_xlim(left=0)

    hist, bins = np.histogram(voter_counts, bins=30)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    axes[2].hist(voter_counts, bins=logbins)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('number of voters $n$')
    axes[2].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    plt.savefig(f'{out_dir}/k-and-h.pdf', bbox_inches='tight')

    plt.close()

    with open(f'results/general-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general, _, max_k, _ = pickle.load(f)

    with open(f'results/general-partial-truncation-results-40-max-10000-trials.pickle', 'rb') as f:
        winners_general_partial, _, _, _ = pickle.load(f)

    with open(f'results/1d-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d, _, _, _, _ = pickle.load(f)

    with open(f'results/1d-partial-truncation-results-40-max-10000-trials-cand-pos.pickle', 'rb') as f:
        winners_1d_partial, _, _, _, _ = pickle.load(f)

    ks = np.arange(3, max_k + 1)
    plt.figure(figsize=(4, 2.5))

    print('\nSimulation winner count means:')
    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full', '1-Euclidean partial')):
        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_means = np.mean(winner_counts, axis=1)
        winner_count_stds = np.std(winner_counts, axis=1)

        print(name, winner_count_means)

        plt.plot(ks, winner_count_means, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)
        plt.fill_between(ks, winner_count_means - winner_count_stds,
                         winner_count_means + winner_count_stds, alpha=0.2, color=color)

    plt.scatter(np.array(candidate_counts) + (np.random.random(len(candidate_counts)) * 0.5) - 0.25,
                np.array(expected_unique_winners),
                label='PrefLib', marker='.', color='red', s=10, zorder=10)

    plt.ylabel('mean # truncation winners')
    plt.xticks([1, 10, 20, 30, 40], [])

    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('plots/mean-winner-counts.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4, 2.5))

    for winners, name in zip((winners_general, winners_general_partial, winners_1d, winners_1d_partial),
                             ('general full', 'general partial', '1-Euclidean full', '1-Euclidean partial')):
        color = 'green' if 'general' in name else 'blue'

        winner_counts = [[len(np.unique(x)) for x in winners[k]] for k in ks]
        winner_count_maxs = np.max(winner_counts, axis=1)
        # winner_count_stds = np.std(winner_counts, axis=1)

        plt.plot(ks, winner_count_maxs, label=name, ls='dashed' if 'partial' in name else 'solid', c=color)

    plt.scatter(np.array(candidate_counts) + (np.random.random(len(candidate_counts)) * 0.5) - 0.25,
                np.array(max_unique_winners), label='PrefLib', marker='.', color='red', s=10, zorder=10)

    plt.plot(ks, np.maximum(1, ks - 1), color='black', ls='dotted', label='theoretical max')
    plt.xlabel('# candidates ($k$)')
    plt.ylabel('max # truncation winners')
    plt.ylim(0, 15)
    plt.xticks([1, 10, 20, 30, 40], [1, 10, 20, 30, 40])
    plt.yticks([1, 5, 10, 15])

    plt.text(5, 6, 'theoretical max', rotation=59)
    plt.savefig('plots/max-winner-counts.pdf', bbox_inches='tight')
    plt.close()


def make_preflib_table():
    # Skip duplicate elections with tons of write-ins
    to_skip = ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']

    descs = {'sl': 'San Leandro, CA', 'pierce': 'Pierce County, WA', 'irish': 'Dublin, Ireland', 'sf': 'San Francisco, CA',
             'takomapark': 'Takoma Park, WA', 'uklabor': 'UK Labour Party', 'aspen': 'Aspen, CO', 'berkley': 'Berkeley, CA',
             'burlington': 'Burlington, VT', 'debian': 'Debian Project', 'ers': 'Anonymous organizations',
             'minneapolis': 'Minneapolis, MN', 'glasgow': 'Glasgow, Scotland', 'apa': 'American Psychological Association',
             'oakland': 'Oakland, CA'}

    with open(f'results/preflib-resampling/all-resampling-results.pickle', 'rb') as f:
        elections, resampled_results, true_results = pickle.load(f)

    collec_counts = defaultdict(int)
    collec_hs = {c: [] for c in descs}
    collec_ns = {c: [] for c in descs}
    collec_ks = {c: [] for c in descs}

    skipped_vote_frac = []
    skipped_counts = []

    print()
    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        if election_name in to_skip:
            continue

        collec_counts[collection] += 1

        collec_hs[collection].append(max(len(b) for b in ballots))

        collec_ns[collection].append(sum(ballot_counts))
        collec_ks[collection].append(len(cand_names))

        skipped_vote_frac.append(skipped_votes / (skipped_votes + sum(ballot_counts)))
        skipped_counts.append(skipped_votes)

        if collec_hs[collection][-1] > collec_ks[collection][-1]:
            print(election_name)
            print(cand_names)
            print(max(ballots, key=lambda x: len(x)))
            print()

    for collec in sorted(collec_counts):
        min_h, max_h = min(collec_hs[collec]), max(collec_hs[collec])
        min_n, max_n = min(collec_ns[collec]), max(collec_ns[collec])
        min_k, max_k = min(collec_ks[collec]), max(collec_ks[collec])

        h_string = min_h if min_h == max_h else f'{min_h}--{max_h}'
        n_string = min_n if min_n == max_n else f'{min_n}--{max_n}'
        k_string = min_k if min_k == max_k else f'{min_k}--{max_k}'

        print(f'\\texttt{{{collec}}} & {descs[collec]} & {collec_counts[collec]} & '
              f'{k_string} & {h_string} & {n_string} \\\\')

    total_votes = sum(x for c in collec_ns for x in collec_ns[c])
    total_skipped = sum(skipped_counts)
    print('PrefLib skipped ballot frac:', total_skipped / (total_votes + total_skipped))


def print_lp_constructions():
    print('\nLP constructions:')

    for file in sorted(glob.glob('results/lp-constructions/*.pickle')):
        with open(file, 'rb') as f:
            res, x_rounded, ballots, counts = pickle.load(f)

        print(file, sum(x_rounded), len(ballots))


if __name__ == '__main__':

    os.makedirs('plots', exist_ok=True)

    plot_truncation_heatmaps()
    plot_combined_small_heatmaps()
    plot_preflib_resampling()
    plot_small_stacked_bars()
    make_preflib_table()
    print_lp_constructions()
