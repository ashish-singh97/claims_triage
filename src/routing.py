"""Cost policy plus exact daily batch capacity allocation."""
import numpy as np
import pandas as pd
QUEUES = ['auto_approve', 'review', 'high_risk']


def costs(probabilities, cfg):
    p = np.asarray(probabilities)
    loss = np.array([0, cfg['denial_loss'], cfg['fwa_loss']])
    return np.column_stack([p @ loss,
        cfg['review_cost'] + p @ (loss*(1-np.array(cfg['review_detection']))),
        cfg['senior_cost'] + p @ (loss*(1-np.array(cfg['senior_detection'])))])


def route_individual(p, cfg, penalty):
    return int(np.argmin(costs(np.asarray(p).reshape(1,3), cfg)[0] + np.array([0,penalty,penalty])))


def route_batch(probabilities, dates, cfg, penalty=0):
    """Greedy positive-benefit allocation; approximate, not globally optimal."""
    c = costs(probabilities, cfg)
    result = np.zeros(len(c), dtype=int)
    for _, indexes in pd.Series(np.arange(len(c))).groupby(pd.Series(dates).str[:10]):
        indexes = indexes.to_numpy()
        candidates = []
        for i in indexes:
            for action in [1,2]:
                benefit = c[i,0]-c[i,action]-penalty
                if benefit > 0:
                    candidates.append((benefit, int(i), action))
        capacity = {1: cfg['review_capacity_per_day'], 2: cfg['senior_capacity_per_day']}
        assigned = set()
        for _, i, action in sorted(candidates, key=lambda r: (-r[0],r[1],r[2])):
            if i not in assigned and capacity[action] > 0:
                result[i] = action
                assigned.add(i)
                capacity[action] -= 1
    return result


def realized_expected_cost(y, actions, cfg):
    """Conditional on synthetic test labels; reviewer effectiveness remains assumed."""
    loss = np.array([0, cfg['denial_loss'], cfg['fwa_loss']])[np.asarray(y)]
    base = np.array([0,cfg['review_cost'],cfg['senior_cost']])[actions]
    detection = np.array([[0,0,0],cfg['review_detection'],cfg['senior_detection']])
    return base + loss*(1-detection[actions, np.asarray(y)])


def select_penalty(p, y, dates, cfg):
    results = []
    for penalty in [0,5,10,20,40,80]:
        actions = route_batch(p, dates, cfg, penalty)
        results.append((float(realized_expected_cost(y,actions,cfg).mean()), penalty))
    return min(results)[1]
