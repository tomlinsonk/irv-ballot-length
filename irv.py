import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


########################
##### DISCRETE IRV #####
########################

def run_discrete_irv(ballots, candidates, h=None, counts=None):
    if h is None:
        h = len(candidates)

    if counts is None:
        counts = [1] * len(ballots)

    ballots = [b[:h] for b in ballots]

    elim_votes = dict()

    active_cands = set(range(len(candidates)))

    while len(active_cands) > 0:
        top_choice_counts = {cand: 0 for cand in active_cands}
        for i in range(len(ballots)):
            b = ballots[i]
            if len(b) > 0:
                top_choice_counts[b[0]] += counts[i]

        if len(top_choice_counts) == 0:
            return elim_votes

        min_cand = min(top_choice_counts, key=top_choice_counts.get)

        active_cands.remove(min_cand)
        elim_votes[min_cand] = top_choice_counts[min_cand]

        for i in range(len(ballots)):
            ballots[i] = ballots[i][ballots[i] != min_cand]

    return elim_votes


##########################
##### CONTINUOUS IRV #####
##########################

def fast_continuous_uniform_plurality(cand_pos):
    cand_idxs = np.argsort(cand_pos)
    ordered_cands = cand_pos[cand_idxs]

    regions = np.concatenate(((0,), (ordered_cands[1:] + ordered_cands[:-1]) / 2, (1,)))
    votes = np.diff(regions)

    return cand_idxs[np.argmax(votes)]


def fast_continuous_uniform_irv(cand_pos):
    k = len(cand_pos)

    cand_idxs = np.argsort(cand_pos)
    ordered_cands = cand_pos[cand_idxs]

    while k > 1:
        regions = np.concatenate(((0,), (ordered_cands[1:] + ordered_cands[:-1]) / 2, (1,)))
        votes = np.diff(regions)
        elim = np.argmin(votes)

        k = k - 1
        cand_idxs = np.delete(cand_idxs, elim)
        ordered_cands = np.delete(ordered_cands, elim)

    return cand_idxs[0]


GO_LEFT = -1
GO_RIGHT = +1


def recurse_on_region(left, right, rank, left_option, right_option, cand_idxs, ordered_cands, regions):
    # Either the midpoint of left_option, right_option is to the left of left, inside the region [left, right], or
    # to the right of right

    # print('Called', left, right, rank, left_option, right_option, cand_idxs, ordered_cands, regions)

    if rank > len(cand_idxs):
        return

    direction = GO_LEFT
    single_recurse = True
    midpoint = None

    if left_option is None and right_option is None:
        return
    elif left_option is None:
        regions[rank][(left, right)] = cand_idxs[right_option]
        direction = GO_RIGHT
    elif right_option is None:
        regions[rank][(left, right)] = cand_idxs[left_option]
    else:
        midpoint = (ordered_cands[left_option] + ordered_cands[right_option]) / 2

        # print(midpoint, left)

        if midpoint < left:
            regions[rank][(left, right)] = cand_idxs[right_option]
            direction = GO_RIGHT
        elif midpoint > right:
            regions[rank][(left, right)] = cand_idxs[left_option]
        else:
            single_recurse = False
            regions[rank][(left, midpoint)] = cand_idxs[left_option]
            regions[rank][(midpoint, right)] = cand_idxs[right_option]

    if single_recurse:
        if direction == GO_RIGHT:
            right_option = right_option + 1 if right_option < len(cand_idxs) - 1 else None
        else:
            left_option = left_option - 1 if left_option > 0 else None
        recurse_on_region(left, right, rank + 1, left_option, right_option, cand_idxs, ordered_cands, regions)
    else:
        new_left_option = left_option - 1 if left_option > 0 else None
        new_right_option = right_option + 1 if right_option < len(cand_idxs) - 1 else None

        recurse_on_region(left, midpoint, rank + 1, new_left_option, right_option, cand_idxs, ordered_cands, regions)
        recurse_on_region(midpoint, right, rank + 1, left_option, new_right_option, cand_idxs, ordered_cands, regions)


def get_continuous_regions(cand_pos):
    """
    Compute the regions at each rank where each candidate is listed by voters in that region at that rank.
    e.g., {r: {(left, right): c}} means that voters in (left, right) rank candidate c and rank r

    :param cand_pos: The positions of all the candidates
    :return: a dict from ranks to dicts from intervals to candidates, such as {rank: {(left, right): candidate}}
    """

    cand_pos = np.array(cand_pos)

    regions = {rank: dict() for rank in range(1, len(cand_pos) + 1)}

    cand_idxs = np.argsort(cand_pos)
    ordered_cands = cand_pos[cand_idxs]

    left = 0
    for i in range(0, len(cand_idxs) - 1):
        right = (ordered_cands[i] + ordered_cands[i + 1]) / 2
        regions[1][(left, right)] = cand_idxs[i]

        recurse_on_region(left, right, 2, None if i == 0 else i - 1, i + 1, cand_idxs, ordered_cands, regions)

        left = right

    regions[1][(left, 1)] = cand_idxs[-1]
    recurse_on_region(left, 1, 2, len(cand_idxs) - 2, None, cand_idxs, ordered_cands, regions)

    return regions


def get_ballot_types(cand_pos, h=None, lengths=None, voter_dsn=None):
    if h is None:
        h = len(cand_pos)

    rank_regions = get_continuous_regions(cand_pos)
    types = sorted(rank_regions[len(cand_pos)].keys(), key=lambda x: x[0])

    ballot_types = []
    votes_per_ballot_type = []

    for i, (left, right) in enumerate(types):
        size = right - left
        reverse_ballot = [rank_regions[len(cand_pos)][left, right]]

        rank = len(cand_pos) - 1

        while rank > 0:
            for (l, r), cand in rank_regions[rank].items():
                if l <= left and r >= right:
                    reverse_ballot.append(cand)
                    break

            rank -= 1

        ballot_types.append(np.array(reverse_ballot)[::-1][:h])

        if lengths is not None:
            ballot_types[-1] = ballot_types[-1][:lengths[i]]

        if voter_dsn is None:
            votes_per_ballot_type.append(size)
        else:
            votes_per_ballot_type.append(voter_dsn.cdf(right) - voter_dsn.cdf(left))

    return ballot_types, votes_per_ballot_type


def run_irv(k, ballot_types, votes_per_ballot_type, cands=None):
    """
    Given a collection of ballot types and the number of voters per ballot type, run IRV
    :param k: the number of candidates
    :param ballot_types: a list of all distinct ballots
    :param votes_per_ballot_type: the number of votes per distinct ballot type
    :return: a dict from candidate indices to elimination votes
    """

    if cands is None:
        cands = range(k)

    elim_votes = dict()
    active_cands = set(cands)

    while len(active_cands) > 0:
        top_choice_sizes = {i: 0 for i in active_cands}
        for i, ballot in enumerate(ballot_types):
            if len(ballot) > 0:
                top_choice_sizes[ballot[0]] += votes_per_ballot_type[i]

        min_cand = min(top_choice_sizes, key=top_choice_sizes.get)

        active_cands.remove(min_cand)
        elim_votes[min_cand] = top_choice_sizes[min_cand]

        for i in range(len(ballot_types)):
            ballot_types[i] = ballot_types[i][ballot_types[i] != min_cand]

    return elim_votes


def run_continuous_uniform_irv(cand_pos, lengths=None, h=None):
    """
    Given candidates at cand_pos and possibly a ballot truncation length h, compute the number of votes each candidate
    has at elimination (the winner is the candidate eliminated with the most votes). Assumes uniform voters over [0, 1]
    :param cand_pos: a np.array of candidate positions in [0, 1]
    :param lengths: the ballots lengths for each voter type, from left to right, or None if all ballots are full
    :param h: the ballot truncation length (or None for full ballots)
    :return: a dict from candidate indices to elimination votes
    """

    ballot_types, votes_per_ballot_type = get_ballot_types(cand_pos, h, lengths)

    return run_irv(len(cand_pos), ballot_types, votes_per_ballot_type)


def run_continuous_weighted_irv(cand_pos, weights, h=None):
    """
    Given candidates at cand_pos and possibly a ballot truncation length h, compute the number of votes each candidate
    has at elimination (the winner is the candidate eliminated with the most votes). Assumes a weighting of ballot types
    (i.e., non-uniform voters)
    :param cand_pos: a np.array of candidate positions in [0, 1]
    :param weights: the number of voters per ballot type, left to right
    :param h: the ballot truncation length (or None for full ballots)
    :return: a dict from candidate indices to elimination votes
    """

    ballot_types, _ = get_ballot_types(cand_pos, h)

    return run_irv(len(cand_pos), ballot_types, weights)


def run_continuous_irv_voter_dsn(cand_pos, voter_dsn, h=None):

    ballot_types, votes_per_ballot_type = get_ballot_types(cand_pos, h, voter_dsn=voter_dsn)

    return run_irv(len(cand_pos), ballot_types, votes_per_ballot_type)


def interactive_plot(n_cands=3):
    # Create subplot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.5)

    # ax = axes[0]

    # ax.invert_yaxis()
    ax.set_ylim(n_cands + 0.5, -0.5)
    ax.set_yticks(range(1, n_cands + 1))

    colors = ["#68c700",
              "#1900aa",
              "#ffb774",
              "#0162bc",
              "#901b00",
              "#ff58ea",
              "#002328",
              "#ff94ab"]


    def fill_between(cands, to_clear):

        for l in to_clear:
            l.remove()

        regions = get_continuous_regions(cands)

        lines = []
        for rank in range(1, len(cands) + 1):
            for (left, right), cand in regions[rank].items():
                lines.append(ax.fill_between([left, right], rank - 0.2, rank + 0.2,
                                             color=colors[cand], alpha=0.5, linewidth=0))

        return lines

    plots = []
    slider_axes = []
    sliders = []

    starting_positions = np.linspace(0, 1, n_cands)

    for i in range(n_cands):
        plots.append(ax.plot([starting_positions[i]], [0], 'o', c=colors[i])[0])
        slider_axes.append(plt.axes([0.25, 0.4 - 0.05 * i, 0.65, 0.02]))
        sliders.append(Slider(slider_axes[-1], f'Candidate {i}', 0.0, 1.0, starting_positions[i]))

    ax.set_title('Rankings')
    ax.set_xlabel('Position')

    to_clear = []

    def update(val, to_clear):
        cand_positions = []
        for i in range(n_cands):
            cand_positions.append(sliders[i].val)
            plots[i].set_xdata([cand_positions[-1]])

        winners = dict()
        for h in range(1, len(cand_positions) + 1):
            elim_votes = run_continuous_uniform_irv(cand_positions, h=h)
            winners[h] = max(elim_votes, key=elim_votes.get)

        ax.set_title(f'{len(set(winners.values()))} winners: {winners}')

        new_to_clear = fill_between(cand_positions, to_clear)
        to_clear.clear()
        to_clear.extend(new_to_clear)

    update(None, to_clear)

    # Call update function when slider value is changed
    for slider in sliders:
        slider.on_changed(lambda x: update(x, to_clear))

    ax.set_xlim(0, 1)

    # display graph
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int)
    args = parser.parse_args()

    if args.k > 8:
        print('Please use at most k=8 (or tweak the code to support more colors)')
        exit(1)

    interactive_plot(args.k)





