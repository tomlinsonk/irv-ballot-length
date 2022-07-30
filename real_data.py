import argparse
import glob
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import yaml
from numpy.random import default_rng
from tqdm import tqdm

from irv import run_irv

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['datadir']


def clean_up_invalid_ballots(ballots, ballot_counts):
    """
    Fix ballots where candidates appear multiple times. Only the first appearance of a candidate is counted
    :param ballots:
    :param ballot_counts:
    :return:
    """

    merged_counts = defaultdict(int)

    for ballot, ballot_count in zip(ballots, ballot_counts):
        clean_ballot, idx = np.unique(ballot, return_index=True)
        clean_ballot = tuple(clean_ballot[np.argsort(idx)])

        merged_counts[clean_ballot] += ballot_count

    ballots, ballot_counts = zip(*merged_counts.items())

    return list(map(np.array, ballots)), ballot_counts


def read_preflib(file_name):
    with open(file_name, 'r') as f:
        n_cands = int(f.readline())

        cand_names = dict()
        for i in range(n_cands):
            split = f.readline().strip().split(',')
            cand_idx = int(split[0])
            cand_names[cand_idx] = ','.join(split[1:])

        n_voters, votes, unique_ballots = map(int, f.readline().strip().split(','))

        ballot_counts = []
        ballots = []
        skipped_votes = 0
        for i in range(unique_ballots):
            line = f.readline().strip()
            split_line = line.split(',')

            # Skip any ballots with ties
            if '{' not in line:
                ballot_counts.append(int(split_line[0]))
                ballots.append(np.array(tuple(map(int, split_line[1:]))))
            else:
                skipped_votes += int(split_line[0])

    ballots, ballot_counts = clean_up_invalid_ballots(ballots, ballot_counts)

    return ballots, ballot_counts, cand_names, skipped_votes


def analyze_election(ballots, ballot_counts, cand_names):
    k = len(cand_names)
    max_ballot_length = max(len(b) for b in ballots)

    winners = []

    majority_winner = get_majority_winner(cand_names, ballots, ballot_counts)

    if majority_winner is None:
        for h in range(1, max_ballot_length + 1):
            truncated_ballots = [b[:h] for b in ballots]
            elim_votes = run_irv(k, truncated_ballots, ballot_counts, cands=cand_names.keys())
            winners.append(max(elim_votes, key=elim_votes.get))
    else:
        winners = [majority_winner] * max_ballot_length

    return winners, majority_winner


def load_all_preflib_elections():
    election_dir = f'{DATA_DIR}/preflib/elections'

    elections = []

    for collection in glob.glob(f'{election_dir}/*'):
        for file_name in glob.glob(f'{collection}/*.toi') + glob.glob(f'{collection}/*.soi'):

            # Skip duplicate elections with write-ins
            if os.path.basename(file_name) in ['ED-00018-00000001.soi', 'ED-00018-00000003.soi']:
                continue

            ballots, ballot_counts, cand_names, skipped_votes = read_preflib(file_name)

            elections.append((
                os.path.basename(collection),
                os.path.basename(file_name),
                ballots,
                ballot_counts,
                cand_names,
                skipped_votes
            ))

    return elections


def get_majority_winner(cand_names, ballots, ballot_counts):
    votes = sum(ballot_counts)

    first_round_votes = {cand: 0 for cand in cand_names}
    for b, bc in zip(ballots, ballot_counts):
        first_round_votes[b[0]] += bc

    if max(first_round_votes.values()) > votes / 2:
        return max(first_round_votes, key=first_round_votes.get)

    return None


def get_plurality_and_second_round_majority_winner(cand_names, ballots, ballot_counts):
    vote_counts = {i: 0 for i in cand_names}
    for i, ballot in enumerate(ballots):
        if len(ballot) > 0:
            vote_counts[ballot[0]] += ballot_counts[i]

    min_cand = min(vote_counts, key=vote_counts.get)
    plurality_winner = max(vote_counts, key=vote_counts.get)

    for b, bc in zip(ballots, ballot_counts):
        if b[0] == min_cand and len(b) > 1:
            vote_counts[b[1]] += bc

    vote_counts.pop(min_cand)

    second_round_leader = max(vote_counts, key=vote_counts.get)

    if vote_counts[second_round_leader] > sum(vote_counts.values()) / 2 and second_round_leader == plurality_winner:
        return plurality_winner

    return None


def resample(ballot_counts, seed=0):
    n = sum(ballot_counts)
    p = np.array(ballot_counts) / n

    rng = default_rng(seed=seed)

    resampled_counts = rng.multinomial(n, pvals=p)

    return resampled_counts


def election_resampling_helper(args):
    (collection, election_name, ballots, ballot_counts, cand_names, skipped_votes), seed = args

    ballot_counts = resample(ballot_counts, seed=seed)

    winners, majority_winner = analyze_election(ballots, ballot_counts, cand_names)

    return collection, election_name, winners, majority_winner


def election_resampling(threads):
    elections = load_all_preflib_elections()
    trials = 10000

    params = ((election, trial) for election in elections for trial in range(trials))

    resampled_results = {(election[0], election[1]): [] for election in elections}
    true_results = {(collection, election_name): analyze_election(ballots, ballot_counts, cand_names)
                    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections}

    with Pool(threads) as pool:
        for collection, election_name, winners, majority_winner in tqdm(pool.imap_unordered(
                election_resampling_helper, params), total=trials*len(elections)):

            resampled_results[collection, election_name].append((winners, majority_winner))

    out_dir = 'results/preflib-resampling'
    os.makedirs(out_dir, exist_ok=True)

    with open(f'{out_dir}/all-resampling-results.pickle', 'wb') as f:
        pickle.dump((elections, resampled_results, true_results), f)


def print_summary_stats():
    elections = load_all_preflib_elections()

    hs = []
    truncation_winners = []

    for collection, election_name, ballots, ballot_counts, cand_names, skipped_votes in elections:
        winners, majority_winner = analyze_election(ballots, ballot_counts, cand_names)

        hs.append(max(len(x) for x in ballots))
        truncation_winners.append(len(np.unique(winners)))

    hs = np.array(hs)
    truncation_winners = np.array(truncation_winners)

    print('Total elections:', len(hs))

    for i, count in enumerate(np.bincount(truncation_winners)):
        print(f'{i} truncation winners: {count} elections')

    print()
    print('Restricting to elections with h <= 5', np.count_nonzero(hs <= 5))
    for i, count in enumerate(np.bincount(truncation_winners[hs <= 5])):
        print(f'{i} truncation winners: {count} elections')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()

    print_summary_stats()
    election_resampling(args.threads)

