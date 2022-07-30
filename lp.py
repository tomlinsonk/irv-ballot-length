import argparse
import glob
import itertools
import os
import pickle
from collections import defaultdict

import numpy as np
import scipy.optimize
from tqdm import tqdm

from irv import run_irv


def construct_all_elim_orders(k):
    order = np.tile(np.arange(k), (k-1, 1))

    for h in range(k - 2):
        order[h, k-1] = h + 1

        order[h, h+1] = k-1

    all_perms = itertools.product(*(itertools.permutations(range(i, k)) for i in range(2, k-1)))

    for perms in all_perms:
        for i, perm in enumerate(perms):
            order[i, i+1:k-1] = perm

        yield order


def get_perms_for_cand(cand, h, k, eliminated):
    cands = np.arange(k)

    # print(cand, h, k, eliminated)

    for cand_pos in range(h+1):

        for prior_cands in itertools.combinations(eliminated, cand_pos):
            remaining_cands = np.setdiff1d(cands, np.concatenate(([cand], prior_cands)))

            for prior_cand_perm in itertools.permutations(prior_cands):
                for fill_perm in itertools.permutations(remaining_cands):
                    yield prior_cand_perm + (cand,) + fill_perm


def construct_lp(elim_order, constraint_gap):
    k = elim_order.shape[1]

    s_k = list(itertools.permutations(range(k)))
    s_k_labels = {perm: i for i, perm in enumerate(s_k)}

    A = []
    b = []

    for h in range(k-1):
        h_order = elim_order[h]
        for i in range(k):
            cand = h_order[i]
            for j in range(i+1, k):
                other = h_order[j]
                # One inequality here.
                # print(f'h={h}, cand {cand} has fewer votes than cand {other} at step {i}')

                # Need all perms giving cand votes at step i
                # They can have votes for cand at position 1, ..., h, as long as all earlier cands have been eliminated

                smaller_var_set = np.array([s_k_labels[perm] for perm in get_perms_for_cand(cand, h, k, h_order[:i])])
                larger_var_set = np.array([s_k_labels[perm] for perm in get_perms_for_cand(other, h, k, h_order[:i])])

                constraint_row = np.zeros(len(s_k))
                constraint_row[larger_var_set] = 1
                constraint_row[smaller_var_set] = -1

                A.append(constraint_row)
                b.append(constraint_gap)

    A = np.array(A)
    b = np.array(b)

    res = scipy.optimize.linprog(np.ones(len(s_k)), -A, -b, method='simplex')

    x_rounded = np.round(res.x).astype(int)

    ballots = []
    counts = []
    for count, perm in zip(x_rounded, s_k):
        if count > 0:
            ballots.append(np.array(perm))
            counts.append(count)

    winners = []
    for h in range(1, k + 1):
        truncated_ballots = [b[:h] for b in ballots]
        elim_votes = run_irv(k, truncated_ballots, counts)
        winners.append(max(elim_votes, key=elim_votes.get))

    success = np.all(A @ x_rounded > 0) and len(np.unique(winners)) == k-1

    return success, res, x_rounded, ballots, counts


if __name__ == '__main__':

    results_dir = 'results/lp-constructions'
    os.makedirs(results_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--make-constructions', action='store_true')
    args = parser.parse_args()

    # For each LP construction, k, constraint_gap, and whether to use all elim orders
    construction_args = [
        (4, 1.1, True),
        (5, 1.01, True),
        (6, 1, True),
        (7, 1, True),
        (8, 4, False),
        (9, 5, False),
        (10, 10, False)
    ]

    if args.make_constructions:
        for k, constraint_gap, use_all_orders in construction_args:
            print(f'Running LP construction for k={k}, constraint_gap={constraint_gap}')

            lower_bound = np.inf
            max_solution = -np.inf

            for i, order in tqdm(enumerate(construct_all_elim_orders(k))):
                success, res, x_rounded, ballots, counts = construct_lp(order, constraint_gap)

                lower_bound = min(lower_bound, sum(res.x))
                max_solution = max(max_solution, sum(res.x))

                if success:
                    with open(f'{results_dir}/k-{k}-constraint-gap-{constraint_gap}-order-{i}.pickle', 'wb') as f:
                        pickle.dump((res, x_rounded, ballots, counts), f)

                if not use_all_orders:
                    print('Skipping other elimination orders')
                    exit(0)

    for fname in sorted(glob.glob(f'{results_dir}/*.pickle')):

        split_fname = os.path.basename(fname).replace('.pickle', '').split('-')
        k = int(split_fname[1])
        constraint_gap = float(split_fname[4])
        order = int(split_fname[6])

        with open(fname, 'rb') as f:
            res, x_rounded, ballots, counts = pickle.load(f)

        print()
        print(f'Construction for k={k}, constraint_gap={constraint_gap}, elimination order {order}')
        print(f'voters: {sum(x_rounded)}, ballot types: {len(ballots)}')

        sums = defaultdict(int)
        for b, c in zip(ballots, counts):
            print(tuple(b), c)
