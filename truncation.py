import argparse
import itertools
import pickle
from multiprocessing import Pool
import numpy as np
import yaml
from numpy.random import default_rng
from tqdm import tqdm
from irv import run_discrete_irv, run_continuous_uniform_irv, run_continuous_weighted_irv, run_continuous_irv_voter_dsn


with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config['resultsdir']



def truncation_1d_helper(args):
    k, trial, weighted, partial_ballots, voter_dsn, cands = args

    rng = default_rng(seed=trial)

    weights = rng.random((k * (k-1)) // 2 + 1)
    lengths = rng.integers(1, k+1, (k * (k-1)) // 2 + 1) if partial_ballots else None

    winners = np.zeros(len(cands), dtype=int)

    full_ballot_elim_order = None

    for h in range(1, len(cands)+1):
        if weighted:
            elim_votes = run_continuous_weighted_irv(cands, weights, h=h)
        elif voter_dsn is not None:
            elim_votes = run_continuous_irv_voter_dsn(cands, voter_dsn, h=h)
        else:
            elim_votes = run_continuous_uniform_irv(cands, lengths=lengths, h=h)

        winners[h-1] = max(elim_votes, key=elim_votes.get)

        if h == len(cands):
            full_ballot_elim_order = sorted(elim_votes, key=elim_votes.get)

    # Reindex winners by their full-ballot elimination order
    reindex = {x: i for i, x in enumerate(full_ballot_elim_order)}
    winners = np.vectorize(reindex.get)(winners)
    cands = cands[full_ballot_elim_order]

    agrees = np.array([int(winners[h-1] == winners[k-1]) for h in range(1, len(cands))])

    return k, trial, agrees, winners, cands


def truncation_1d_experiment(n_threads, weighted=False, partial_ballots=False, voter_dsn=None, name='', cand_samples=None):
    max_k = 40
    ks = np.arange(1, max_k + 1)
    n_trials = 10_000

    assert not (partial_ballots and weighted), 'No support for weighted partial ballots'
    assert not (partial_ballots and voter_dsn), 'No support for custom distribution partial ballots'
    assert not (weighted and voter_dsn), 'Cant do weighting and custom dsn'

    rng = default_rng(seed=0)

    params = ((k, trial, weighted, partial_ballots, voter_dsn, rng.random(k) if cand_samples is None else rng.choice(cand_samples[k], size=k)) for k in ks for trial in range(n_trials))

    agree_counts = np.zeros((max_k, max_k))

    all_winners = {k: np.zeros((n_trials, k)) for k in ks}
    all_cands = {k: np.zeros((n_trials, k)) for k in ks}

    with Pool(n_threads) as pool:
        for k, trial, agrees, winners, cands in tqdm(pool.imap_unordered(truncation_1d_helper, params), total=len(ks)*n_trials):
            agree_counts[k-1, :len(agrees)] += agrees
            all_winners[k][trial] = winners
            all_cands[k][trial] = cands

    agree_frac = agree_counts / n_trials

    with open(f'results/1d-{"weighted-" if weighted else ""}{"partial-" if partial_ballots else ""}{name}truncation-results-'
              f'{max_k}-max-{n_trials}-trials-cand-pos.pickle', 'wb') as f:
        pickle.dump((all_winners, all_cands, agree_frac, max_k, n_trials), f)


def truncation_general_helper(args):
    k, n, trial, partial_ballots = args

    rng = default_rng(seed=trial)

    cands = np.arange(k)
    ballots = rng.permuted(np.tile(cands, n).reshape(n, k), axis=1)

    if partial_ballots:
        ballots = [b[:rng.integers(1, k+1)] for b in ballots]

    winners = np.zeros(len(cands), dtype=int)

    for h in range(1, len(cands)+1):
        elim_votes = run_discrete_irv(ballots, cands, h=h)
        winners[h-1] = max(elim_votes, key=elim_votes.get)

    agrees = np.array([int(winners[h-1] == winners[k-1]) for h in range(1, len(cands))])

    return k, trial, agrees, winners


def truncation_general_experiment(n_threads, partial_ballots=False):
    max_k = 40
    n = 1000
    ks = np.arange(1, max_k + 1)
    n_trials = 10_000

    params = ((k, n, trial, partial_ballots) for k in ks for trial in range(n_trials))

    agree_counts = np.zeros((max_k, max_k))

    all_winners = {k: np.zeros((n_trials, k)) for k in ks}

    with Pool(n_threads) as pool:
        for k, trial, agrees, winners in tqdm(pool.imap_unordered(truncation_general_helper, params), total=len(ks) * n_trials):
            agree_counts[k - 1, :len(agrees)] += agrees
            all_winners[k][trial] = winners

    agree_frac = agree_counts / n_trials

    with open(f'results/general-{"partial-" if partial_ballots else ""}truncation-results-{max_k}-max-{n_trials}-trials.pickle', 'wb') as f:
        pickle.dump((all_winners, agree_frac, max_k, n_trials), f)


def truncation_partial_construction(k):
    assert k > 3
    candidate_ballots = [[[i] + list(range(i)) for _ in range(2 * (k - 2) + 1)] for i in range(k)]

    candidate_ballots[1].append([1, 0])
    for i in range(k):
        if i > 0:
            candidate_ballots[i].append([i] + list(range(i)))

        idx = 0
        for j in range(i + 2, k):
            candidate_ballots[i][idx] += [j]
            idx += 1

            candidate_ballots[i][idx] += [j]
            idx += 1

        if i <= k - 3:
            candidate_ballots[i][idx] += [i + 2]

    ballots = list(itertools.chain.from_iterable(candidate_ballots))

    return ballots


def load_cand_samples(cand_dsn_file, h):
    with open(cand_dsn_file, 'rb') as f:
        all_winners, all_cands, agree_frac, max_k, n_trials = pickle.load(f)

    winner_positions = dict()
    for k in range(1, max_k + 1):
        winner_positions[k] = all_cands[k][np.arange(n_trials)[:, None], all_winners[k].astype(int)][:, h-1]

    return winner_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()

    truncation_1d_experiment(args.threads, weighted=False, partial_ballots=False)
    truncation_1d_experiment(args.threads, weighted=False, partial_ballots=True)

    truncation_general_experiment(args.threads)
    truncation_general_experiment(args.threads, partial_ballots=True)



